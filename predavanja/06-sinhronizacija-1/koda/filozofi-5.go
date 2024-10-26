// 5 filozofov
// s ključavnicami omejimo uporabo vilic
// pravilno delovanje, vsi pobirajo vilice v istem vrstnem redu

package main

import (
	"flag"
	"fmt"
	"sync"
	"time"
)

var wg sync.WaitGroup
var fork [5]sync.Mutex

func session(dishes int, id int) {
	defer wg.Done()

	fmt.Println("Philosopher", id, "approached.")
	forkId1 := id
	forkId2 := (id + 1) % 5
	if id == 4 {
		forkId1, forkId2 = forkId2, forkId1
	}

	dish := 0
	for dish < dishes {
		dish++
		fmt.Println("Philosopher", id, "is thinking.")
		time.Sleep(100 * time.Millisecond)
		fork[forkId1].Lock()
		fmt.Println("Philosopher", id, "took fork", forkId1, ".")
		time.Sleep(100 * time.Millisecond)
		fork[forkId2].Lock()
		fmt.Println("Philosopher", id, "took fork", forkId2, ".")
		time.Sleep(100 * time.Millisecond)
		fmt.Println("Philosopher", id, "is eating", dish, ".")
		time.Sleep(100 * time.Millisecond)
		fork[forkId1].Unlock()
		fork[forkId2].Unlock()
		fmt.Println("Philosopher", id, "put down the forks.")
		time.Sleep(100 * time.Millisecond)
	}
	fmt.Println("Philosopher", id, "left.")
}

func main() {
	// preberemo argumente
	dPtr := flag.Int("d", 20, "# of dishes")
	flag.Parse()
	// razdelimo delo med gorutine in jih zaženemo
	timeStart := time.Now()
	wg.Add(5)
	for id := 0; id < 5; id++ {
		go session(*dPtr, id)
	}
	// gorutine pridružimo
	wg.Wait()
	timeElapsed := time.Since(timeStart)
	fmt.Println("Čas:", timeElapsed)
}
