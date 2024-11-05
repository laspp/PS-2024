/*
Paket xkcd omogoča branje opisov stripov iz strani xkcd.com, ki so na voljo v formatu JSON
*/
package xkcd

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

const Url = "https://xkcd.com/"

// struktura Comic hrani podatke o posameznem stripu
// Id: zaporedna številka stripa
// Title: naslov stripa
// Transcript: prepis besedila iz stripa in opis scene
// Tooltip: napis, ki se pokaže pri stripu
type Comic struct {
	Id         int    `json:"num"`
	Title      string `json:"title"`
	Transcript string `json:"transcript"`
	Tooltip    string `json:"alt"`
}

// funkcija prebere podatke iz spletne strani xkcd za strip s številko id
// vrne podatkovno strukturo Comic, ki vsebuje podatke o stripu
func FetchComic(id int) (Comic, error) {
	client := &http.Client{Timeout: 5 * time.Minute}
	var url string
	if id == 0 {
		url = Url + "/info.0.json"
	} else {
		url = Url + fmt.Sprintf("%d", id) + "/info.0.json"
	}
	req, err := http.NewRequest("GET", url, nil)

	data := Comic{}

	if err != nil {
		return data, fmt.Errorf("http request error: %v", err)
	}

	resp, err := client.Do(req)

	if err != nil {
		return data, fmt.Errorf("http err: %v", err)
	} else {
		defer resp.Body.Close()
	}

	if resp.StatusCode != http.StatusOK {
	} else {
		if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
			return data, fmt.Errorf("json error: %v", err)
		}
	}
	return data, nil

}
