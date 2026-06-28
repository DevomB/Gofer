package main

// Sample is one training record (M10/M11 export schema).
type Sample struct {
	BoardHash uint64    `json:"board_hash"`
	MoveNum   int       `json:"move_num"`
	Policy    []float32 `json:"policy"`
	PolicyOpp []float32 `json:"policy_opp,omitempty"`
	ToPlay    Color     `json:"to_play"`
	Ownership []float32 `json:"ownership,omitempty"`
	ScorePDF  []float64 `json:"score_pdf,omitempty"`
	ScoreCDF  []float64 `json:"score_cdf,omitempty"`
}
