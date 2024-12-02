// čas UTC in monotoni čas
// v izpisu sta združena čas UTC in monotoni čas (m)
// 		Time start  : 2023-11-12 11:01:22.846808047 +0100 CET m=+2.001286487
// 		Time end    : 2023-11-12 11:01:23.847060075 +0100 CET m=+3.001538559
// 		Time elapsed: 1.000252072s
//
// 		razlika UTC: 1.000252028 s, razlika m: 1.000252072 s

package main

import (
	"fmt"
	"time"
)

func main() {

	time.Sleep(2 * time.Second)

	timeStart := time.Now()
	time.Sleep(1 * time.Second)
	timeEnd := time.Now()
	timeElapsed := timeEnd.Sub(timeStart)

	fmt.Printf("Time start  : %v\n", timeStart)
	fmt.Printf("Time end    : %v\n", timeEnd)
	fmt.Printf("Time elapsed (UTC)      : %vs\n", float64(timeEnd.UnixNano()-timeStart.UnixNano())/1e9)
	fmt.Printf("Time elapsed (monotonic): %v\n", timeElapsed)
}
