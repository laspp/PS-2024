// Pregrada
// ključavnica, sodi in lihi obhodi zank

package main

import (
	"flag"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

var wg sync.WaitGroup
var goroutines int
var g int = 0
var phaseGlobal int = 0 // sodi, lihi obhod zanke
var lock sync.Mutex

func barrier(id int, printouts int) {

	var phaseLocal int = 0

	defer wg.Done()

	for i := 0; i < printouts; i++ {

		// operacije v zanki
		time.Sleep(time.Duration(rand.Intn(10)) * time.Millisecond)
		fmt.Println("Gorutine", id, "printout", i)

		// pregrada - začetek
		lock.Lock()
		g++
		lock.Unlock()
		phaseLocal = 1 - phaseLocal
		if g < goroutines {
			for phaseGlobal != phaseLocal {
			}
		} else {
			g = 0
			phaseGlobal = phaseLocal
		}
		// pregrada - konec
	}
}

func main() {
	// preberemo argumente
	gPtr := flag.Int("g", 4, "# of goroutines")
	pPtr := flag.Int("p", 5, "# of printouts")
	flag.Parse()

	goroutines = *gPtr

	// zaženemo gorutine
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go barrier(i, *pPtr)
	}
	// počakamo, da vse zaključijo
	wg.Wait()
}
