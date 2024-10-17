// pozdrav
// 		module load Go
// 		srun --reservation=psistemi --tasks=1 --cpus-per-task=2 go run pozdrav-1.go
//
// 		go build pozdrav-1.go
// 		srun --reservation=psistemi --tasks=1 --cpus-per-task=2 ./pozdrav-1

package main

import (
	"fmt"
	"time"
)

const printouts = 10

func hello() {
	var i int
	for i = 0; i < printouts; i++ {
		fmt.Print("hello world ")
		time.Sleep(time.Millisecond)
	}
}

func main() {
	hello()
	fmt.Println()
}
