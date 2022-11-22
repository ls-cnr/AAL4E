<h1>Documentazione Ble Scan</h1>



<h2>Obiettivo</h2>

L'obiettivo è quello di andare a determinare la posizione di un utente all'interno di un ambiente utilizzando dei dispositivi beacon bluetooth low energy.
Più nello specifico, ciò che bisogna andare a definire sono le coordinate nell'ambiente dei beacon e il loro measured power.
Quando nell'applicazione viene premuto il tasto "Start Scan", si avvierà la scansione  bluetooth. Se c'è un match tra gli UID dei beacon rilevati e dei beacon presenti nella struttura *dict_pair* allora verrà qui inserito il valore rssi. 
Viene inoltre definita una costante intera *Interval* che viene utilizzata per lanciare un thread ogni *Interval* secondi.
Il thread si occuperà di dare in output la posizione dell'utente, nel paragrafo delle funzioni viene approfondito questo meccanismo.

<h2>Librerie Android utilizzate e gestione permessi</h2> 



<h2>Classi</h2>

Classe ***Pair***: rappresenta l'oggetto formato da: 

- device, in cui viene memorizzato l'UID del beacon;
- device_rssi, che è un intero in cui viene memorizzato il valore rssi ottenuto durante la scansione bluetooth.

Classe ***DistPair***: rappresenta  l'oggetto formato da:

- device, in cui viene memorizzato l'UID del beacon;
- dist, che è un double in cui è memorizzata la distanza dell'utente dal beacon.

Classe ***Coord***: questa classe contiene le coordinate X e Y relative ad un beacon. Verrà poi utilizzata nella struttura *map_coord_beacon*.



<h2>Strutture dati utilizzate</h2>

***dict_pair*** : è una HashMap che contiene come chiave una stringa che rappresenta l'UID del beacon del quale vogliamo andare a memorizzare il valore rssi durante la scansione; come valore ha una lista di Integer in cui verranno inseriti i valori rssi.

***map_coord_beacon***: è una HashMap che ha, anche lei, come chiave la stringa che rappresenta l'UID del beacon e come valore ha un oggetto Coord, cioè le coordinate relative al beacon.

***list_position*** : è una lista di oggetti Coord in cui vengono memorizzate le coordinate dell'utente ogni Interval secondi. Questa viene resettata nel momento in cui viene effettuata una nuova scansione.



<h2>Funzioni e la loro interazione</h2>

Si parte dalla funzione *onBtnScan*: questa viene chiamata quando dall'applicazione viene premuto il pulsante con id: btnScan; si occupa di andare a richiamare la funzione *scanLeDevice* e di settare il contenuto del bottone.

La funzione *scanLeDevice* si occupa di andare a verificare se ci sono i permessi definiti all'interno della funzione *checkBtPermissions* e di invocare la funzione di libreria *startScan* su un oggetto di tipo *BluetoothLeScanner* passando come parametro una funzione callback *insRssiDictScanCallback*.
Inoltre viene invocata la funzione *startRepeatingTask* che si occuperà di invocare il thread.

La funzione callback *insRssiDictScanCallback* definisce ciò che viene effettuato quando durante la scansione viene trovato un ble, cioè se l'UID è presente nella struttura *dict_pair* allora viene aggiunto alla sua lista il valore rssi. Inoltre viene settato il valore della text view.

La funzione Runnable *mHandlerTask* richiama al suo interno *insertSmallerRssiList*, successivamente resettare le liste di *dict_pair*.

La funzione *insertSmallerRssiList* definisce una lista di oggetti Pair chiamata *average_position* un *temp_dict_pair* che rappresenta una copia di *dict_pair*. 
Attraverso un ciclo for che itera su ogni lista di *temp_dict_pair* vado a fare la media degli rssi (se la lista non è vuota), creo l'oggetto Pair e lo aggiungo a *average_position*. Successivamente, se *average_position* non è vuota creo una lista di oggetti *DistPair* che sarà uguale all'output della funzione *convertRssiToDist*. Inoltre viene anche invocata la funzione *get_position*, che restituirà in output un oggetto *Coord*. 
Infine aggiungo l'oggetto *Coord* alla lista *list_position* e lo utilizzo per settare la text view relativa alle coordinate dell'utente.

La funzione *convertRssiToDist* prende come parametro la lista *average_position* e restituisce in output una lista di oggetti *DistPair*. Scorrendo la lista *average_position* vado a convertire il valore del rssi relativo ad un beacon in una distanza, quindi creo l'oggetto DistPair e lo aggiungo alla lista che verrà restituita in output. Inoltre la lista viene prima ordinata secondo il valore della distanza, in modo tale da avere per primi i beacon che hanno una distanza minore.

La funzione *getMesPow* viene utilizzata all'interno di *convertRssiToDist* e si occupa semplicemente di restituire il measured power relativo all'UID del beacon passato come parametro.

Le funzioni *getPosition* e *getPosition2* rappresentano due diversi modi per implementare la triangolazione dell'utente conoscendo le distanze dai beacon e le loro coordinate.

 

<h2>Installazione</h2>

Aprire cartella progetto con Android Studio, collegare smartphone al pc (usb o wifi). Una volta riconosciuto il telefono premere su pulsante play.
Bisogna avere scaricato le sdk relative al proprio sistema Android.





<h2>Diagramma funzioni</h2>

![](/home/pablo/MEGA/Università/Tirocinio/Diagramma funzioni.drawio.png)
