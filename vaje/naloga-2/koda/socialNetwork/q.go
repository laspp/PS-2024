/*
Paket socialNetwork nudi funkcije za ustvarjanje, zagon in ustavitev generatorja zahtevkov za indeksiranje objav
*/
package socialNetwork

import (
	_ "embed"
	"math/rand"
	"strings"
	"time"
)

// Definiramo velikost vrste
const max_queue_length = 10000

// Preberemo objave in ključne besede iz datoteke
//
//go:embed modrosti.txt
var content string

// Podatkovna struktura Task, ki predstavlja posamezni zahtevek
type Task struct {
	Id   uint64
	Data string
}

// Podatkovna struktura Q, ki vlkjučuje vse potrebne spremenljivke za generiranje zahtevkov
type Q struct {
	N                  uint64
	TaskChan           chan Task
	quit               chan bool
	rnd                *rand.Rand
	listOfFortunes     []string
	delay              int
	averageQueueLength float64
	maxQueueLength     int
}

// Ustvari nov generator, inicializira vse podatkovne struture
// S parametrom delay nastavimo zakasnitev med posameznimi zahtevki
func (load *Q) New(delay int) {
	load.listOfFortunes = strings.Split(content, "\n%\n")
	load.rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
	load.TaskChan = make(chan Task, max_queue_length)
	load.quit = make(chan bool)
	load.delay = delay
}

// Zaženemo generator
func (load *Q) Run() {
	var newTask Task
	for {
		select {
		case <-load.quit:
			close(load.TaskChan)
			return
		default:
			i := load.rnd.Intn(len(load.listOfFortunes))
			newTask = Task{Id: uint64(i), Data: load.listOfFortunes[i]}
			load.N++
			load.TaskChan <- newTask
			queueN := len(load.TaskChan)
			load.averageQueueLength += ((float64(queueN) - load.averageQueueLength) / float64(load.N))
			load.maxQueueLength = max(load.maxQueueLength, queueN)
			for d := 0; d < load.delay; d++ {
				//zakasnitev do naslednjega zahtevka
			}
		}
	}
}

// Vrne povprečno dolžino vrste v %
func (load Q) GetAverageQueueLength() float64 {
	return load.averageQueueLength / max_queue_length * 100
}

// Vrne največjo dolžino vrste v %
func (load Q) GetMaxQueueLength() float64 {
	return float64(100*load.maxQueueLength) / float64(max_queue_length)
}

// Ustavimo generator
func (load *Q) Stop() {
	load.quit <- true
	close(load.quit)
}

// Vrne true, če je vrsta prazna
func (load Q) QueueEmpty() bool {
	return len(load.TaskChan) == 0
}
