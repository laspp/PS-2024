// primer uporabe paketa storage
// operacije CRUD kličemo lokalno

package main

import (
	"api/storage"
	"fmt"
)

func main() {
	var reply struct{}
	var replyMap map[string]storage.Todo

	// ustvarimo shrambo
	store := storage.NewTodoStorage()

	// pripravimo strukture, ki jih uporabljamo kot argumente
	lecturesCreate := storage.Todo{Task: "predavanja", Completed: false}
	readAll := storage.Todo{Task: "", Completed: false}

	// ustvarimo zapis
	fmt.Print("1. Put: ")
	if err := store.Put(&lecturesCreate, &reply); err != nil {
		panic(err)
	}
	fmt.Println("done")

	// preberemo en zapis
	fmt.Print("2. Get 1: ")
	replyMap = make(map[string]storage.Todo)
	if err := store.Get(&lecturesCreate, &replyMap); err != nil {
		panic(err)
	}
	fmt.Println(replyMap, ": done")

	// preberemo vse zapise
	fmt.Print("4. Get *: ")
	replyMap = make(map[string]storage.Todo)
	if err := store.Get(&readAll, &replyMap); err != nil {
		panic(err)
	}
	fmt.Println(replyMap, ": done")
}
