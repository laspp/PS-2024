# 1. Domača naloga: Analiza zbirke stripov xkcd

**Rok za oddajo: 19. 11. 2024**

Znana zbirka stripov [xkcd](https://xkcd.com/) omogoča programski dostop do informacij o posameznem stripu. Podatki so na voljo v formatu [JSON](https://www.json.org/). Podatki o najnovejšem stripu so na voljo na naslovu [https://xkcd.com/info.0.json](https://xkcd.com/info.0.json), do starejših pa dostopamo s pomočjo zaporedne številke. Na primer, strip številka 1 dobimo na naslovu [https://xkcd.com/1/info.0.json](https://xkcd.com/1/info.0.json). Trenutno je objavljenih 3007 stripov.

![xkcd API](https://imgs.xkcd.com/comics/api.png)

Za vsak strip so na voljo naslednji podatki:
- Mesec (*month*)
- Zaporedna številka (*num*)
- Povezava (*link*) 
- Leto (*year*)
- Novica (*news*)
- Varni naslov (*safe_title*)
- Prepis stripa (*transcript*)
- Dodatno pojasnilo (*alt*)
- Povezava do slike (*img*)
- Naslov (*title*)
- Dan (*day*)

Nekatera polja se ne uporabljajo in so vedno prazna, npr. *news* in *link*. Primer izpisa JSON za strip [številka 1](https://xkcd.com/1/):
```json
{
   "month":"1",
   "num":1,
   "link":"",
   "year":"2006",
   "news":"",
   "safe_title":"Barrel - Part 1",
   "transcript":"[[A boy sits in a barrel which is floating in an ocean.]]\nBoy: I wonder where I'll float next?\n[[The barrel drifts into the distance. Nothing else can be seen.]]\n{{Alt: Don't we all.}}",
   "alt":"Don't we all.",
   "img":"https://imgs.xkcd.com/comics/barrel_cropped_(1).jpg",
   "title":"Barrel - Part 1",
   "day":"1"
}
```

V repozitoriju na povezavi [github.com/laspp/PS-2024/vaje/naloga-1/koda/xkcd](https://github.com/laspp/PS-2024/tree/main/vaje/naloga-1/koda/xkcd) najdete modul xkcd, ki vsebuje funkcijo: 
```Go
func FetchComic(id int) (Comic, error)
```
Funkcija naredi poizvedbo na spletni stran [xkcd](https://xkcd.com/) in izlušči podatke o stripu številka `id`. Če uporabite `0` kot argument, potem funkcija izlušči podatke o najnovejšem stripu. Funkcija nato vrne podatkovno strukturo tipa `Comic`, ki vsebuje naslednja polja:
```Go
type Comic struct {
	Id         int    `json:"num"`
	Title      string `json:"title"`
	Transcript string `json:"transcript"`
	Tooltip    string `json:"alt"`
}
```
Če funkcija stripa ne najde, potem vrne prazno strukturo. V primeru, da pride pri dostopu do spletne strani do težav, potem funkcija vrne neničelno vrednost za napako `error`.

Paket si namestite tako, da v mapi, kjer boste pisali vaš program, zaženete ukaze
```Bash
$ go mod init <ime_vasega_modula>
$ go get github.com/laspp/PS-2024/vaje/naloga-1/koda/xkcd
```
V vaši datoteki .go potem uvozite modul tako, da v sekcijo `import` dodate vrstico `github.com/laspp/PS-2024/vaje/naloga-1/koda/xkcd`

```Go
import (
    "fmt"
    "github.com/laspp/PS-2024/vaje/naloga-1/koda/xkcd"
)
```

Vaša naloga je napisati **sočasen program**, ki mu lahko preko argumenta ukazne vrstice navedete koliko delavcev (gorutin) naj uporabi pri analizi stripov. Program naj nato med delavce **enakomerno** razdeli vse stripe, ki so trenutno objavljeni in **sočasno prešteje vse besede, ki so dolge vsaj 4 znake**, v podatkih o vseh stripih. **Podatke o frekvencah besed hranite v slovarju.** Pazite na kritične sekcije in na ustreznih mestih uporabite kanale ali druge konstrukte za sinhronizacijo. Pri štetju besed uporabite vsebine polj `Title`, `Transcript` in `Tooltip`. 

> [!NOTE]
> Pri starejših stripih polje `Transcript` vsebuje tudi vsebino polja `Tooltip`. Pri novejših pa je polje `Transcript` prazno. Da se izognete dvakratnemu štetju besed pri stripih, ignorirajte polje `Tooltip`, kjer polje `Transcript` ni prazno.

Pred štetjem besed iz besedil **odstranite vsa ločila in posebne znake** ter pretvorite vse **črke v male**. Pomagajte si s paketom [`strings`](https://pkg.go.dev/strings).

Primer predprocesiranja za strip 1:
```
barrel part 1 a boy sits in a barrel which is floating in an ocean boy i wonder where i ll float next the barrel drifts into the distance nothing else can be seen alt don t we all
```
Primer preštetih besed dolgih vsaj 4 znake za strip 1:
```
barrel, 3
float, 1
part, 1
which, 1
drifts, 1
next, 1
into, 1
sits, 1
else, 1
distance, 1
floating, 1
ocean, 1
wonder, 1
seen, 1
where, 1
```

**Na koncu besede sortirajte glede na frekvenco in izpišite 15 najpogostejših besed in njihovih frekvenc za celoten nabor stripov**. Pomagajte si s paketom [`sort`](https://pkg.go.dev/sort). Nalogo do roka oddajte preko [spletne učilnice](https://ucilnica.fri.uni-lj.si/mod/assign/view.php?id=37715).


