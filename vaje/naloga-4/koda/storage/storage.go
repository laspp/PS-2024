// paket storage
// 		enostavna shramba nalog, zgrajena kot slovar
//		strukturo Todo pri protokolu rest prenašamo v obliki sporočil JSON, zato pri opisu dodamo potrebne označbe
//		strukturi TodoStorage definiramo metode
//			odtis funkcije (argumenti, vrnjene vrednosti) je tak, da jezik go zna tvoriti oddaljene klice metod

package api

import (
	"errors"
	"sync"
)

type Todo struct {
	Task      string `json:"task"`
	Completed bool   `json:"completed"`
	Commited  bool   `json:"commited"`
}

type TodoStorage struct {
	dict map[string]Todo
	// ključavnica za bralne in pisalne dostope do shrambe
	mu sync.RWMutex
}

var ErrorNotFound = errors.New("not found")

func NewTodoStorage() *TodoStorage {
	dict := make(map[string]Todo)
	return &TodoStorage{
		dict: dict,
	}
}

func (tds *TodoStorage) Get(todo *Todo, dict *map[string]Todo) error {
	tds.mu.RLock()
	defer tds.mu.RUnlock()
	if todo.Task == "" {
		for k, v := range tds.dict {
			(*dict)[k] = v
		}
		return nil
	} else {
		if val, ok := tds.dict[todo.Task]; ok {
			(*dict)[val.Task] = val
			return nil
		}
		return ErrorNotFound
	}
}

func (tds *TodoStorage) Put(todo *Todo, ret *struct{}) error {
	tds.mu.Lock()
	defer tds.mu.Unlock()
	todo.Commited = false
	tds.dict[todo.Task] = *todo
	return nil

}

func (tds *TodoStorage) Commit(todo *Todo, ret *struct{}) error {
	tds.mu.Lock()
	defer tds.mu.Unlock()
	if t, ok := tds.dict[todo.Task]; ok {
		t.Commited = true
		tds.dict[todo.Task] = t
		return nil
	}
	return ErrorNotFound

}
