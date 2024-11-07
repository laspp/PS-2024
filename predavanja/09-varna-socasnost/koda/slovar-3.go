// Varna sočasnost
//		uporabimo slovar iz paketa sync, sync.Map

package main

import (
	"flag"
	"fmt"
	"sync"
	"time"
)

var wg sync.WaitGroup

func writeToMap(id int, steps int, dict *sync.Map) {
	defer wg.Done()
	dict.Store(id, 0)
	for i := 0; i < steps; i++ {
		val, _ := dict.Load(id)
		dict.Store(id, val.(int)+1) // zahtevanje tipa (type assertion)
	}
}

func readFromMap(id int, steps int, dict *sync.Map) {
	defer wg.Done()
	for i := 0; i < steps; i++ {
		_, _ = dict.Load(id)
	}
}

func main() {
	gwPtr := flag.Int("gw", 1, "# of writing goroutines")
	grPtr := flag.Int("gr", 1, "# of reading goroutines")
	sPtr := flag.Int("s", 100, "# of read or write steps")
	flag.Parse()

	var dict sync.Map
	records := *gwPtr
	if *grPtr > records {
		records = *grPtr
	}
	for i := 0; i < records; i++ {
		dict.Store(i, 0)
	}

	timeStart := time.Now()
	for i := 0; i < *gwPtr; i++ {
		wg.Add(1)
		go writeToMap(i, *sPtr, &dict)
	}
	for i := 0; i < *grPtr; i++ {
		wg.Add(1)
		go readFromMap(i, *sPtr, &dict)
	}
	wg.Wait()

	fmt.Print("dict: map[")
	dict.Range(
		func(k, v interface{}) bool {
			fmt.Print(k, ":", v, " ")
			return true
		})
	fmt.Println("\b] time:", time.Since(timeStart))
}
