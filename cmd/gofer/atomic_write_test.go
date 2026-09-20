package main

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
	"time"
)

// A report must never be observable half-written: the gating stage reuses any
// report file it finds, so a truncated one is permanent (it fails every resume).
func TestArenaReportWriteIsAtomic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "batch-01.json")

	if err := writeFileAtomic(path, []byte(`{"wins_challenger":7}`), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `{"wins_challenger":7}` {
		t.Fatalf("content = %q", got)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatal("temp file left behind after a successful write")
	}

	// Overwriting replaces the whole file rather than truncating in place.
	if err := writeFileAtomic(path, []byte(`{"a":1}`), 0644); err != nil {
		t.Fatal(err)
	}
	if got, _ = os.ReadFile(path); string(got) != `{"a":1}` {
		t.Fatalf("after overwrite content = %q", got)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatal("temp file left behind after an overwrite")
	}
}

func TestArenaReportWriteFailurePropagates(t *testing.T) {
	// Parent directory does not exist: the temp write fails, nothing is created.
	path := filepath.Join(t.TempDir(), "missing", "report.json")
	if err := writeFileAtomic(path, []byte("x"), 0644); err == nil {
		t.Fatal("expected an error writing into a missing directory")
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatal("temp file should not exist")
	}
}

// --------------------------------------------------------------- retry loop

var errTransient = errors.New("simulated sharing violation")

func isSimulated(err error) bool { return errors.Is(err, errTransient) }

func TestReportRenameRetriesTransientErrors(t *testing.T) {
	calls := 0
	rename := func(_, _ string) error {
		calls++
		if calls < 3 {
			return errTransient
		}
		return nil
	}
	if err := renameLoop(rename, isSimulated, "a", "b", 10, time.Millisecond); err != nil {
		t.Fatalf("expected success after retries, got %v", err)
	}
	if calls != 3 {
		t.Fatalf("calls = %d, want 3", calls)
	}
}

func TestReportRenameStopsOnPermanentError(t *testing.T) {
	permanent := errors.New("no such file")
	calls := 0
	rename := func(_, _ string) error {
		calls++
		return permanent
	}
	err := renameLoop(rename, isSimulated, "a", "b", 10, time.Millisecond)
	if !errors.Is(err, permanent) {
		t.Fatalf("err = %v, want the permanent error", err)
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1: a permanent error must not be retried", calls)
	}
}

func TestReportRenameGivesUpAfterAllAttempts(t *testing.T) {
	calls := 0
	rename := func(_, _ string) error {
		calls++
		return errTransient
	}
	err := renameLoop(rename, isSimulated, "a", "b", 4, time.Millisecond)
	if !errors.Is(err, errTransient) {
		t.Fatalf("err = %v, want the transient error surfaced after giving up", err)
	}
	if calls != 4 {
		t.Fatalf("calls = %d, want 4", calls)
	}
}

// The classifier is deliberately Windows-only: elsewhere a rename failure is
// real and retrying would just delay the report.
func TestReportRenameTransientClassification(t *testing.T) {
	sharing := &os.LinkError{Op: "rename", Err: syscall.Errno(32)}
	notFound := &os.LinkError{Op: "rename", Err: syscall.Errno(2)}

	if runtime.GOOS != "windows" {
		if isTransientRenameErr(sharing) {
			t.Fatal("no rename error should be treated as transient off Windows")
		}
		return
	}
	if !isTransientRenameErr(sharing) {
		t.Fatal("ERROR_SHARING_VIOLATION should be retried")
	}
	if isTransientRenameErr(notFound) {
		t.Fatal("ERROR_FILE_NOT_FOUND is permanent and must not be retried")
	}
	if isTransientRenameErr(nil) {
		t.Fatal("nil is not an error")
	}
}
