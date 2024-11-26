# Posredovanje sporočil

Pri komunikaciji med procesi v omrežju se običajno naslanjamo na komunikacijski sklad, ki ga nudi operacijski sistem. Tipično se poslužujemo protokola Ethernet, ki deluje na povezovalni plasti komunikacijskega protokola in ga nadgrajuje protokol IP (angl. Internet Protocol) na internetni plasti. Transportna plast nadgrajuje internetno in tu najdemo protokola kot sta TCP (angl. Transmission Control Protocol) in UDP (angl. User Datagram Protocol). Po potrebi se nad transportno plastjo nahajajo še dodatne plasti, ki nudijo specifične funkcionalnost za dano aplikacijo (šifriranje, avtentikacija, ...).

Protokol TCP je namenjen zanesljivemu prenosu podatkov med dvema procesoma. Skrbi za odpravljanje napak pri prenosu, razrešuje podvajanje in izgubo paketov ter nadzira pretok. Uporablja povezavni način komuniciranja, kjer procesa najprej vzpostavita povezavo, šele nato lahko začneta med sabo izmenjevati uporabne podatke. S protokolom TCP pridobimo na zanesljivosti prenosov na račun manjše prepustnosti. 

V primeru, da zanesljivost ni pomembna, ampak želimo čim večjo prepusntost in čim manjšo zakasnitev se lahko poslužimo protokola UDP. Ta uporablja brezpovezavni način komunikacije in omogoča hiter prenos podatkov, vendar na račun zanesljivosti. Ne omogoča nadzora pretoka, prav tako ne zagotavlja dostave ali pravilnega vrstnega reda sporočil. Za razliko od protokola TCP tu nimamo opravka s podatkovnim tokom, ki ga protokol samodejno razbije na podatkovne pakete, ampak aplikacija sama pripravi pakete omejene dolžine (angl. Datagram) in jih pošlje prejemniku.

## Primer aplikacije
Delovanje protokolov TCP in UDP bomo prikazali na simulirani igri telefončki. Glavni proces pošlje sporočilo naslednjemu v vrsti. Ta sporočilo prejme, ga dopolne/spremeni in posreduje naslednjemu. Zadnji proces v vrsti prejme sporočilo, ga dopolne in vrne glavnemu procesu. Ta ga na koncu izpiše, s tem se izmenjava sporočil zaključi.

![Telefon](telefon.png)

### Komunikacija preko protokola TCP

Vsak proces mora delovati kot klient in strežnik hkrati. Določiti moramo vrata na katerih bo poslušal za prihajajoče sporočilo in vrata na katera bo svoje sporočilo poslal. Da je primer enostven, bomo vse procese zagnali na enem vozlišču. S tem se izognemo nastavljanju naslovov IP za vsak proces.

Vsak proces dobi svoj `id`, hkrati pa mora tudi vedeti, koliko procesov je bilo zagnanih. Komunikacija poteka po zaporednih vratih od izhodiščnih naprej. Izhodiščna vrata nastavimo preko argumenta ukazne vrstice, prav tako `id` procesa in skupno število vseh procesov. 
```Go
package main

import (
	"flag"
	"fmt"
	"net"
	"time"
)

type message struct {
	data   []byte
	length int
}

var N int
var id int

func checkError(err error) {
	if err != nil {
		panic(err)
	}
}
func receive(addr *net.TCPAddr) message {
	// Poslušamo
	listener, err := net.ListenTCP("tcp", addr)
	checkError(err)
	fmt.Println("Telefon", id, "posluša na", addr)
	conn, err := listener.Accept()
	checkError(err)
	defer conn.Close()
	buffer := make([]byte, 1024)
	// Preberemo sporočilo
	mLen, err := conn.Read(buffer)
	checkError(err)
	fmt.Println("Telefon", id, "prejel:", string(buffer[:mLen]))
	// Vrnemo sporočilo
	rMsg := message{}
	rMsg.data = append(rMsg.data, buffer[:mLen]...)
	rMsg.length = mLen
	return rMsg
}

func send(addr *net.TCPAddr, msg message) {
	var err error
	retry := 5
	var conn *net.TCPConn
	// Odpremo povezavo, lahko je potrebnih več poskusov, če niso še vsi pripravljeni
	for retry != 0 {
		conn, err = net.DialTCP("tcp", nil, addr)
		retry--
		time.Sleep(time.Second / 2)
		if err == nil {
			break
		}
	}
	checkError(err)
	defer conn.Close()
	fmt.Println("Telefon", id, "se je povezal na", addr)
	// Pripravimo sporočilo
	sMsg := fmt.Sprint(id) + "-"
	sMsg = string(msg.data[:msg.length]) + sMsg
	_, err = conn.Write([]byte(sMsg))
	checkError(err)
	fmt.Println("Telefon", id, "poslal sporočilo", sMsg, "telefonu na naslovu", addr)

}

func main() {
	// Preberi argumente
	portPtr := flag.Int("p", 9000, "# start port")
	idPtr := flag.Int("id", 0, "# process id")
	NPtr := flag.Int("n", 2, "total number of processes")
	flag.Parse()

	rootPort := *portPtr
	id = *idPtr
	N = *NPtr
	basePort := rootPort + id
	nextPort := rootPort + ((id + 1) % N)
	// Ustvari potrebne mrežne naslove
	localAddr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("localhost:%d", basePort))
	checkError(err)
	remoteAddr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("localhost:%d", nextPort))
	checkError(err)

	// Izmenjava sporočil
	if id == 0 {
		send(remoteAddr, message{})
		rMsg := receive(localAddr)
		fmt.Println(string(rMsg.data[:rMsg.length]) + "0")
	} else {
		rMsg := receive(localAddr)
		send(remoteAddr, rMsg)
	}

}
```

### Komunikacija preko protokola UDP

Za razliko od komunikacije preko TCP, tukaj posamezni proces ne more vedeti, kdaj je naslednji proces v vrsti začel poslušati na vratih. V kolikor je sporočilo poslano preden je prejemnik pripravljen, se bo sporočilo izgubilo. Da zagotovimo pripravljenost vseh procesov uvedemo gorutino `heartBeat`, ki poskrbi, da glavni proces `id==0` počaka na prejem obvestil od vseh ostalih procesov, da so pripravljeni, preden pošlje sporočilo prvemu. 

```Go
package main

import (
	"flag"
	"fmt"
	"net"
	"time"
)

type message struct {
	data   []byte
	length int
}

var start chan bool
var stopHeartbeat bool
var N int
var id int

func checkError(err error) {
	if err != nil {
		panic(err)
	}
}
func receive(addr *net.UDPAddr) message {
	// Poslušamo
	conn, err := net.ListenUDP("udp", addr)
	checkError(err)
	defer conn.Close()
	fmt.Println("Telefon", id, "posluša na", addr)
	buffer := make([]byte, 1024)
	// Preberemo sporočilo
	mLen, err := conn.Read(buffer)
	checkError(err)
	fmt.Println("Telefon", id, "prejel sporočilo:", string(buffer[:mLen]))
	// Vrnemo sporočilo
	rMsg := message{}
	rMsg.data = append(rMsg.data, buffer[:mLen]...)
	rMsg.length = mLen
	return rMsg
}

func send(addr *net.UDPAddr, msg message) {
	// Odpremo povezavo
	conn, err := net.DialUDP("udp", nil, addr)
	checkError(err)
	defer conn.Close()
	// Pripravimo sporočilo
	sMsg := fmt.Sprint(id) + "-"
	sMsg = string(msg.data[:msg.length]) + sMsg
	_, err = conn.Write([]byte(sMsg))
	checkError(err)
	fmt.Println("Telefon", id, "poslal sporočilo", sMsg, "telefonu na naslovu", addr)
	// Ustavimo heartbeat servis
	stopHeartbeat = true
}

func heartBeat(addr *net.UDPAddr) {

	if id != 0 {
		// Ostali javljajo procesu 0, da so živi
		conn, err := net.DialUDP("udp", nil, addr)
		checkError(err)
		defer conn.Close()
		beat := [1]byte{byte(id)}
		for !stopHeartbeat {
			_, err = conn.Write(beat[:])
			time.Sleep(time.Second)
		}
	} else {
		// Posluša samo 0
		conn, err := net.ListenUDP("udp", addr)
		checkError(err)
		defer conn.Close()
		beat := make([]byte, 1)
		clients := make(map[byte]bool)
		for !stopHeartbeat {
			_, err = conn.Read(beat)
			checkError(err)
			fmt.Println("Telefon", id, "prejel utrip:", beat[:], len(clients))
			clients[beat[0]] = true
			// Če so se vsi javili zaključimo
			if len(clients) == N-1 {
				start <- true
				return
			}
		}
	}
}

func main() {
	// Preberi argumente
	portPtr := flag.Int("p", 9000, "# start port")
	idPtr := flag.Int("id", 0, "# process id")
	NPtr := flag.Int("n", 2, "total number of processes")
	flag.Parse()

	rootPort := *portPtr
	id = *idPtr
	N = *NPtr
	basePort := rootPort + 1 + id
	nextPort := rootPort + 1 + ((id + 1) % N)

	// Ustvari potrebne mrežne naslove
	rootAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("localhost:%d", rootPort))
	checkError(err)

	localAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("localhost:%d", basePort))
	checkError(err)

	remoteAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("localhost:%d", nextPort))
	checkError(err)

	// Ustvari kanal, ki bo signaliziral, da so vsi procesi pripravljeni
	start = make(chan bool)

	// Zaženemo heartbeat servis, ki čaka, na javljanje vseh udeleženih procesov
	stopHeartbeat = false
	go heartBeat(rootAddr)

	// Izmenjava sporočil
	if id == 0 {
		<-start
		send(remoteAddr, message{})
		rMsg := receive(localAddr)
		fmt.Println(string(rMsg.data[:rMsg.length]) + "0")
	} else {
		rMsg := receive(localAddr)
		send(remoteAddr, rMsg)
	}
}
```

**Skripta za zagon na gruči Arnes**
Spodnja skripta s pomočjo nizov poslov SLURm zažene poljubno število "telefonov" v vrsti.

```Bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array=0-4
#SBATCH --reservation=fri
#SBATCH --output=telefon-%a.txt

path=./telefonUDP

module load Go
go build $path/telefon.go
srun telefon -p 9000 -id $SLURM_ARRAY_TASK_ID -n $SLURM_ARRAY_TASK_COUNT
```

Skripto v mapi `koda` zaženete z ukazom
```Bash
$ sbatch run_telefon.sh 
```

V skripti nastavimo parametre posla. Pomembno je, da nastavimo število računskih vozlišč na 1 (`--nodes=1`). S stikalom `--array` nastavimo število podposlov, ki se zaženejo in jih oštevilčimo. V zgornjem primeru zaženemo 5 poslov z oznakami 0-4. Ipisi, ki jih naredijo posamezne instance programa telefon se shranijo v ločene datoteke, vsaka instanca v svojo. To obnašanje nastavimo s stikalom `--output`. Okoljski spremenljivki `$SLURM_ARRAY_TASK_ID` in `$SLURM_ARRAY_TASK_COUNT` se samodejno inicializirata ob zagonu posla in hranita zaporedno številko podposla v nizu in skupno število zagnanih podposlov. 