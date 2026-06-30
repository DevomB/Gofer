package main

// Sample is one self-play training record.
type Sample struct {
	BoardHash uint64    `json:"board_hash"`
	MoveNum   int       `json:"move_num"`
	Policy    []float32 `json:"policy"`
	PolicyOpp []float32 `json:"policy_opp,omitempty"`
	ToPlay    Color     `json:"to_play"`
	Value     float32   `json:"value,omitempty"`
	Komi      float64   `json:"komi,omitempty"`
	Ownership []float32 `json:"ownership,omitempty"`
	ScorePDF  []float64 `json:"score_pdf,omitempty"`
	ScoreCDF  []float64 `json:"score_cdf,omitempty"`
}
