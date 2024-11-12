## Metode in vmesniki

# Metode

Go ni zasnovan kot objektno orientiran jezik, vsebuje pa nekatere konstrukte objektno orientiranih jezikov. En od teh konstruktov so metode.

Metoda je funkcija, ki vsebuje dodaten argument to je prejemnik. Podatkovni tip prejemnika je lahko struktura, možni so pa tudi drugi podatkovni tipi, ki jih definiramo znotraj paketa. Podatkovni tipi definirani izven trenutnega paketa niso dovoljeni. Sintaksa metode je naslednja:

```Go
func (receiver type) methodName(parameter list) returnType {
}
```

Primer uporabe metod v Go:

```Go
/*
Program prikazuje uporabo metod v programskem jeziku Go
*/
package main

import (
	"fmt"
)

// Definirmao strukturo Student
type Student struct {
	name    string
	surname string
	id      string
	year    int
}

// Funkcija za izpis imena in priimka študenta
func displayNameSurname(s Student) {
	fmt.Printf("Ime: %s, Priimek: %s\n", s.name, s.surname)
}

// Metoda za izpis imena in priimka študenta
func (s Student) displayNameSurname() {
	fmt.Printf("Ime: %s, Priimek: %s\n", s.name, s.surname)
}

// Metoda, ki vrača letnik študija
func (s Student) getYear() int {
	return s.year
}

// Metoda, ki nastavi letnik študija
// Uporabimo kazalec za prejemnika
func (s *Student) setYear(year int) {
	s.year = year
}

func main() {

	// Ustvarimo novega študenta
	student1 := Student{name: "Janez", surname: "Novak", id: "63230000", year: 1}

	// Pokličemo funkcijo
	displayNameSurname(student1)

	// Pokličemo metode
	student1.displayNameSurname()
	fmt.Println(student1.getYear())
	student1.setYear(2)
	fmt.Println(student1.getYear())
}
```
[metode.go](koda/metode.go)

Primarna razloga za vpeljavo in uporabo metod sta naslednja:

- Go ni objektno usmerjen programski jezik in ne podpira razredov. Metode na tipih so torej način za dosego obnašanja, podobnega razredom. Metode omogočajo logično združevanje obnašanja, povezanega z določenim tipom, podobno kot razredi. V zgornjem primeru programa so vsa obnašanja, povezana s tipom `Student`, lahko združena z ustvarjanjem metod z uporabo sprejemnika tipa `Student`. 

- Metode z istim imenom in različnimi podatkovnimi tipi argumentov so dovoljene, medtem ko funkcije z istimi imeni niso dovoljene. Recimo, da bi imeli v zgornji kodi dodatno strukturo `Profesor`, ki bi vsebovala polji `name` in `surname`. Za to strukturo lahko ustvarimo metodo `displayNameSurname`, ki že obstaja. Funkcije z istim imenom pa ne bi mogli.

Prejemnik metode je lahko vrednost ali kazalec. Če uporabimo vrednost kot v primeru metode `getYear()`, spremembe, ki jih ta metoda naredi ne bodo vidne zunaj metode. Če uporabimo kazalec na strukturo kot prejemnika pa se bodo spremembe ohranile tudi po zaključku metode. Primer take metode je `setYear(year int)`.

## Vmesniki

V go je vmesnik (angl. Interface) nabor podpisov metod. Ko za nek podatkovni tip pripravimo definicijo vseh metod v vmesniku, pravimo, da podatkovni tip implementira vmesnik. To je zelo podobno svetu objektnega programiranja. Vmesnik določa, katere metode mora tip imeti, in tip določi, kako so metode implementirane. Preko vmesnikov go podpira polimorfizem v kodi.

Sintaksa vmesnika je naslednja:
```Go
type interfaceName interface {
        method1() type
        method2() type
        ...
}
```
Primer uporabe vmesnikov v go:
```Go
/*
Program prikazuje uporabo vmesnikov v programskem jeziku Go
*/
package main

import (
	"fmt"
    "math"
)

// Definiramo nov vmesnik za izračun površine lika
type areaCalculator interface {
    area() float32
}

// Definirmao strukturo za krog
type circle struct {
    radius float32
}

// Definiramo strukturo za pravokotnik
type rect struct {
    width float32
    heigth float32
}

// Metoda za izračun površine kroga
func (c circle) area() float32{
    return c.radius*c.radius*math.Pi

}

// Metoda ta izračun površine pravokotnika
func (r rect) area() float32{
    return r.width * r.height
}

func main() {

    // Ustvarimo nekaj likov
    circle1, circle2, circle3 := circle{10}, circle{3}, circle{8}
    rect1, rect2, rect3 := rect{2,2}, rect{3,8}, rect{10,10}

    // Ustvarimo rezino vmesnikov areaCalculator in vanjo damo vse like, ki implementirajo vmesnik
    shapes := []areaCalculator{circle1, circle2, circle3, rect1, rect2, rect3}
    
    // Izračunajmo skupno površino vseh likov v rezini
    totalArea := 0.0
    for _, v := range shapes {
		totalArea = totalArea + v.area()
	}
	fmt.Printf("Skupna površina likov %f", totalArea)
```
[vmesniki.go](koda/vmesniki.go)

Rezino `shapes` smo napolnili z različnimi liki. Ker vsi liki implementirajo vmesnik `areaCalculator` lahko zelo enostavno izračunamo skupno površino vseh likov, ne glede na tip lika. V kolikor bi v prihodnosti dodali v kodo še kak tip lika, katerega površino hočemo računati to ne bi zahtevalo večjih sprememb v kodi.

### Prazen vmesnik

Vmesnik, ki nima nobene metode, se imenuje prazen vmesnik. Predstavljen je z `interface{}`. Ker prazen vmesnik nima nobenih metod, vsi podatkovni tipi implementirajo prazen vmesnik. Ker se prazni vmesniki v go pogosto uporabljajo imamo zanje definirano sopomenko `any`

Primer:
```Go
/*
Program prikazuje uporabo praznih vmesnikov v programskem jeziku go
*/
package main

import (
	"fmt"
)

// Funkcija kot argument prejme prazen vmesnik
// Namesto interface{} lahko pišemo tudi any
func display(x interface{}) {
	fmt.Printf("Tip: %T, Vrednost: %v\n", x, x)
}

func main() {
	// Uporaba funkcije display nad različnimi podatkovnimi tipi
	s := "Porazdeljeni sistemi"
	display(s)
	i := 42
	display(i)
	rect := struct {
		width  float32
		heigth float32
	}{
		width:  5,
		heigth: 4,
	}
	display(rect)
}
```
[prazen-vmesnik.go](koda/prazen-vmesnik.go)
