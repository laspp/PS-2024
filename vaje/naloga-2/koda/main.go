/*
Primer uporabe paketa socialNetwork
*/
package main

import (
	"fmt"
	//Uvozimo paket socialNetwork (ustrezno popravite glede na ime projekta)
	"naloga2/socialNetwork"
	"time"
)

func main() {
	// Definiramo nov generator
	var producer socialNetwork.Q
	// Inicializiramo generator. Parameter določa zakasnitev med zahtevki
	producer.New(5000)

	start := time.Now()
	// Delavec, samo prevzema zahtevke
	go func() {
		for {
			<-producer.TaskChan
		}
	}()
	// Zaženemo generator
	go producer.Run()
	// Počakamo 5 sekund
	time.Sleep(time.Second * 5)
	// Ustavimo generator
	producer.Stop()
	// Počakamo, da se vrsta sprazni
	for !producer.QueueEmpty() {
	}
	elapsed := time.Since(start)
	// Izpišemo število generiranih zahtevkov na sekundo
	fmt.Printf("Processing rate: %f MReqs/s\n", float64(producer.N)/float64(elapsed.Seconds())/1000000.0)
	// Izpišemo povprečno dolžino vrste v čakalnici
	fmt.Printf("Average queue length: %.2f %%\n", producer.GetAverageQueueLength())
	// Izpišemo največjo dolžino vrste v čakalnici
	fmt.Printf("Max queue length %.2f %%\n", producer.GetMaxQueueLength())
}
