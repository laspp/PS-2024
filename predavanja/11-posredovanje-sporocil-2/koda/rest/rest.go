// Komunikacija po protokolu HTTP (REST)
//
// 		strežnik ustvari in vzdržuje shrambo nalog TodoStorage
//		odjemalec nad shrambo izvaja operacije CRUD
//
// zaženemo strežnik
// 		go run *.go
// zaženemo enega ali več odjemalcev
//		go run *.go -s [ime strežnika] -p [vrata]
// za [ime strežnika] in [vrata] vpišemo vrednosti, ki jih izpiše strežnik ob zagonu
//
// pri uporabi SLURMa lahko s stikalom --nodelist=[vozlišče] določimo vozlišče, kjer naj se program zažene
//
// odjemalca lahko nadomestimo z orodjem curl iz ukazne vrstice:
// 		1. curl -i -X POST -H "Content-Type: application/json" -d "{\"task\": \"predavanja\", \"completed\": false}" http://localhost:9876/todos
// 		2. curl -i -X GET http://localhost:9876/todos/predavanja
// 		3. curl -i -X POST -H "Content-Type: application/json" -d "{\"task\": \"vaje\", \"completed\": false}" http://localhost:9876/todos
// 		4. curl -i -X GET http://localhost:9876/todos
// 		5. curl -i -X PUT -H "Content-Type: application/json" -d "{\"task\": \"predavanja\", \"completed\": true}" http://localhost:9876/todos/predavanja
// 		6. curl -i -X DELETE http://localhost:9876/todos/vaje
// 		7. curl -i -X GET http://localhost:9876/todos

package main

import (
	"flag"
	"fmt"
)

func main() {
	// preberemo argumente iz ukazne vrstice
	sPtr := flag.String("s", "", "server URL")
	pPtr := flag.Int("p", 9876, "port number")
	flag.Parse()

	// zaženemo strežnik ali odjemalca
	url := fmt.Sprintf("%v:%v", *sPtr, *pPtr)
	if *sPtr == "" {
		Server(url)
	} else {
		Client("http://" + url + "/todos/")
	}
}
