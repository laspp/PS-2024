// paket storage
// 		enostavna shramba nalog, zgrajena kot slovar
//		strukturo Todo pri protokolu rest prenašamo v obliki sporočil JSON, zato pri opisu dodamo potrebne označbe
//		strukturi TodoStorage definiramo metode
//			odtis funkcije (argumenti, vrnjene vrednosti) je tak, da jezik go zna tvoriti oddaljene klice metod

package storage

import (
	"errors"
	"sync"
)

type Todo struct {
	Task      string `json:"task"`
	Completed bool   `json:"completed"`
}

type TodoStorage struct {
	dict map[string]Todo
	mu   sync.RWMutex
}

var ErrorNotFound = errors.New("not found")

func NewTodoStorage() *TodoStorage {
	dict := make(map[string]Todo)
	return &TodoStorage{
		dict: dict,
	}
}

func (tds *TodoStorage) Create(todo *Todo, ret *struct{}) error {
	tds.mu.Lock()
	defer tds.mu.Unlock()
	tds.dict[todo.Task] = *todo
	return nil
}

func (tds *TodoStorage) Read(todo *Todo, dict *map[string]Todo) error {
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

func (tds *TodoStorage) Update(todo *Todo, ret *struct{}) error {
	tds.mu.Lock()
	defer tds.mu.Unlock()
	if _, ok := tds.dict[todo.Task]; ok {
		tds.dict[todo.Task] = *todo
		return nil
	}
	return ErrorNotFound
}

func (tds *TodoStorage) Delete(todo *Todo, ret *struct{}) error {
	tds.mu.Lock()
	defer tds.mu.Unlock()
	if _, ok := tds.dict[todo.Task]; ok {
		delete(tds.dict, todo.Task)
		return nil
	}
	return ErrorNotFound
}
