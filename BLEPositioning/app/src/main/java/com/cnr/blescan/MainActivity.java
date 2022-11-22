package com.cnr.blescan;

import static android.Manifest.permission.BLUETOOTH_SCAN;

import android.Manifest;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanResult;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class Pair{

    String device; //uid device
    int device_rssi;

    public Pair(String a, int b){
        this.device = a;
        this.device_rssi = b;
    }
    @Override
    public String toString(){
        return this.device + " " + this.device_rssi;
    }
}

class DistPair{
    String device; //uid device
    Float dist;    // distanza del beacon dal device

    public DistPair(String a, Float b) {
        this.device = a;
        this.dist = b;
    }

    @Override
    public String toString(){
        return this.device + " " + this.dist;
    }


}


class ListPair{ // questa classe dovrebbe essere cancellabile
    //Viene utilizzata per andare a memorizzare i timestamp delle medie degli rssi
    public List<Pair> list_pair = new ArrayList<Pair>(); //lista che contiene le coppie

    @Override
    public String toString(){
        String temp = "";
        for(Pair ele : list_pair){
            temp += ele.device + " " + ele.device_rssi + "\n";
        }
        return temp;
    }



}

class Coord{
    //Questa classe viene utilizzata come secondo parametro del dizionario in cui
    //c'è come chiave UID beacon e valore le due coordinate
    Integer x;
    Integer y;

    public Coord(int x, int y){
        this.x = x;
        this.y = y;
    }
    @Override
    public String toString(){
        return this.x + " " + this.y;
    }
}


public class MainActivity extends AppCompatActivity {
    public List<Pair> list_obj = new ArrayList<Pair>(); //lista che contiene le coppie
    public List<ListPair> list_rssi_medi = new ArrayList<ListPair>(); //lista che contiene i valori medi degli rssi
    //Questa sotto è la struttura dizionario che si occupa di conservare l'UID del dispositivo
    //e una lista che contiene gli rssi letti per quello specifico dispositivo.
    //Utilizzo la classe interna anonima per l'inizializzazione
    public Map<String,List<Integer> > dict_pair = new HashMap<String,List<Integer>>(){
        {
            put("D8:59:D1:1A:53:81", new ArrayList<Integer>()); //mi_band senza scotch
            put("3C:A3:08:AE:1E:C0", new ArrayList<Integer>());
            put("3C:A3:08:AE:24:01", new ArrayList<Integer>());
            put("3C:A3:08:AE:24:02", new ArrayList<Integer>());
            put("3C:A3:08:AE:18:F2", new ArrayList<Integer>());
            put("DB:9D:52:29:F4:84", new ArrayList<Integer>()); //mi band con scotch
        }
    };
    public Map<String, Coord> map_coord_beacon = new HashMap<String, Coord>(){
        {
            put("D8:59:D1:1A:53:81", new Coord (1,2)); //mi_band senza scotch
            put("3C:A3:08:AE:1E:C0", new Coord(1,2));
            put("3C:A3:08:AE:24:01", new Coord(1,2));
            put("3C:A3:08:AE:24:02", new Coord(1,2));
            put("3C:A3:08:AE:18:F2", new Coord(1,2));
            put("DB:9D:52:29:F4:84", new Coord(1,2)); //mi band con scotch
        }
    };
    List<Coord> list_position = new ArrayList<>();
    private BluetoothAdapter mBluetoothAdapter = null;
    private BluetoothLeScanner mBluetoothLeScanner = null;

    public static final int REQUEST_BT_PERMISSIONS = 0;
    public static final int REQUEST_BT_ENABLE = 1;

    private boolean mScanning = false;
    private Handler mHandler = null;

    private Button btnScan = null;
    private TextView textv = null;
    private TextView textv_coordinate = null;
    private final static int INTERVAL = 1000 * 60; //600 secondi cioè 10 minuti



    Runnable mHandlerTask = new Runnable()
    {
        /*
        In questa funzione andiamo ad utilizzare la struttura dict_pair, in cui sono presenti
        le coppie chiave-valore uid-listaRssi
        */
        @Override
        public void run() {
            //creo la stringa da stampare nel log
            String a = ": \nuid,rssi\n";

            // invoco la funzione che si occupa di memorizzare rssi minore e medio
            insertSmallerRssiList();

            for(String uid: dict_pair.keySet()) { //ciò che segue è utile per debug
                for(Integer rssi:dict_pair.get(uid)){
                    a+=uid + "," + rssi + "\n";
                }
            }
            //Log.v("prova",a); //stampo la stringa creata in precedenza
            a="";
            //reinizializzo il dizionario
            dict_pair = new HashMap<String,List<Integer>>(){
                {
                    put("3C:A3:08:AE:24:02", new ArrayList<Integer>());
                    put("D8:59:D1:1A:53:81", new ArrayList<Integer>()); //mi_band senza scotch
                    put("3C:A3:08:AE:1E:C0", new ArrayList<Integer>());
                    put("3C:A3:08:AE:24:01", new ArrayList<Integer>());
                    put("DB:9D:52:29:F4:84", new ArrayList<Integer>()); //mi band con scotch
                    put("3C:A3:08:AE:18:F2", new ArrayList<Integer>());

                }
            };
            mHandler.postDelayed(mHandlerTask, INTERVAL);
        }
    };

    void startRepeatingTask()
    {
        Log.v("BLE", "startRepeatingTask");
        mHandlerTask.run();
    }

    void stopRepeatingTask()
    {
        Log.v("BLE", "stopRepeatingTask");
        mHandler.removeCallbacks(mHandlerTask);
    }
    private ScanCallback mLeScanCallback =
            new ScanCallback() {

                @Override
                public void onScanResult(int callbackType, final ScanResult result) {
                    super.onScanResult(callbackType, result);
                    //Log.v("BLE", "return onScanResult" + ActivityCompat.checkSelfPermission(MainActivity.this,  Manifest.permission.BLUETOOTH_CONNECT));
                    /*if(Build.VERSION.SDK_INT > Build.VERSION_CODES.R){
                        if (ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
                            // TODO: Consider calling
                            //    ActivityCompat#requestPermissions
                            // here to request the missing permissions, and then overriding
                            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                            //                                          int[] grantResults)
                            // to handle the case where the user grants the permission. See the documentation
                            // for ActivityCompat#requestPermissions for more details.
                            Log.v("BLE", "error permission onScanResult");
                            checkBtPermissions();
                            return;
                        }
                    }
                    else{
                        if(ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.BLUETOOTH) != PackageManager.PERMISSION_GRANTED){
                            Log.v("BLE", "error permission onScanResult android 11");
                            checkBtPermissions();
                            return ;
                        }
                    }*/


                    //Log.v("BLE", result.getDevice().getAddress() + "\t" + result.getRssi() + "\t" + result.getDevice().getAlias() + "\t" + result.getTxPower());
                    String dev_addr = result.getDevice().getAddress().toString();
                    //// => vedere se può servire result.getTxPower()

                    Integer b = result.getRssi();
                    if(dict_pair.containsKey(dev_addr)){
                        Log.v("prova",  "ins: " + result.getDevice().getAddress() + "\t" + result.getRssi());
                        dict_pair.get(dev_addr).add(b); //basta effettuare inserimento del valore nella apposita chiave
                                                        //poiché ho già la chiave
                        textv.setText(dev_addr);

                    }

                }

                @Override
                public void onScanFailed(int errorCode) {
                    super.onScanFailed(errorCode);
                    Log.v("BLE", "error");
                }
            };

    public void insertSmallerRssiList(){
        /*Scorrendo sul ciclo for vado a leggere la lista degli rssi,
         * prendo il minimo e lo memorizzo all'interno della lista min_position.
         * Mentre nella lista average_position vado a memorizzare i valori medi.
         * Il problema adesso è che su queste due liste che ho ottenuto devo andare a calcolare le distanze.
         * Questo lo faccio in un'altra funzione che possiamo chiamare anche da qui dentro.
         * */

        List<Pair> min_position = new ArrayList<Pair>();
        List<Pair> average_position = new ArrayList<Pair>();
        Map<String,List<Integer> > temp_dict_pair = new HashMap<String,List<Integer>>(){}; //copia di dict_pair
        temp_dict_pair.putAll(dict_pair); //lavoro su una copia del dizionario, poiché si vanno aggiungendo sempre nuove coppie

        Log.v("l_r_m", "Ho creato temp_list");

        for(String temp_uid:temp_dict_pair.keySet()){
            String uid = temp_uid;
            Integer min = -200;
            if(temp_dict_pair.get(temp_uid).size() > 0){
                for(Integer temp_rssi : temp_dict_pair.get(temp_uid)){
                    if(min < temp_rssi){
                        min = temp_rssi;
                    }

                } //per ogni uid ottengo il valore minimo, a questo punto devo andare ad aggiungere la coppia
                Pair temp_pair = new Pair(uid, min);
                min_position.add(temp_pair);
            }

        }

        for(String uid: temp_dict_pair.keySet()) {
            int size = temp_dict_pair.get(uid).size();
            int sum = 0;
            Log.v("min_average_list", uid);
            if (size > 0) {
                for (Integer rssi_temp : temp_dict_pair.get(uid)) {
                    Log.v("min_average_list", rssi_temp.toString());
                    sum += rssi_temp;
                    Log.v("l_r_m", "sum = " + sum);
                }
                Log.v("min_average_list", "size: " + size);

                int media = sum / size;
                Pair temp_pair = new Pair(uid, media);
                average_position.add(temp_pair);
            }
        }
        //stampare a schermo le due liste ottenute
        /*Log.v("min_average_list", "\n\nmin_position:");
        for(Pair p: min_position) { //scorrendo sul ciclo for devo:
            Log.v("min_average_list", p.device + " " + p.device_rssi);
        }
        Log.v("min_average_list", "average_position:");
        for(Pair p: average_position) { //scorrendo sul ciclo for devo:
            Log.v("min_average_list", p.device + " " + p.device_rssi);
        }*/

        /* A questo punto dovrei andare a calcolare la posizione passando come parametro una lista,
        così posso utilizzarla sia per gli rssi medi che per i minimi.
        Questa funzione si chiamerà convertRssiToDist, prenderà come parametro una lista, in
        modo da poterla utilizzare sia sui min che average.
        * */
        if(min_position.size() > 0){
            List<DistPair> min_pos_converted = convertRssiToDist(min_position);

            // Adesso che ho ottenuto la lista con le distanze posso andare a implementare la trilaterazione
            Coord coord = get_position(min_pos_converted);
            Log.v("cooooooordinate",coord.toString());
            textv_coordinate.setText(coord.x + "<=x | y=>" + coord.y);
            min_position.clear();
            average_position.clear();
        }

    }

    public List<DistPair> convertRssiToDist(List<Pair> l){
        /* Questa funzione si occupa di calcolare la distanza leggendo la lista delle coppie.
        Viene utilizzata la struttura dizionario map_coord poiché qui sono state dichiarate le coordinate x,y dei beacon
        * */
        List<Pair> l_pair = new ArrayList<>(l);
        List<DistPair> temp_dist_pair = new ArrayList<>();
        Integer N = 2;
        for(Pair p : l_pair){
            Integer mes_pow = getMesPow(p.device);
            String uid = p.device;
            Integer dev_rssi = p.device_rssi;
            Log.v("dev_rssi", ""+p.device_rssi);
            Float a = (float)(mes_pow-dev_rssi);
            Float b = (float)(10*N);
            Float c = a/b;
            Float dist = (float)(Math.pow(10,c));
            //p.device_rssi = 10^((mes_pow-p.device_rssi)/(10N));
            DistPair temp = new DistPair(uid,dist);
            Log.v("dev_rssi", temp.device + "=> " + temp.dist + "m");
            temp_dist_pair.add(temp);
        }
        Collections.sort(temp_dist_pair, new Comparator<DistPair>() {
            public int compare(DistPair o1, DistPair o2) {
                // compare two instance of `Score` and return `int` as result.
                return o1.dist.compareTo(o2.dist);
                //return o2.getScores().get(0).compareTo(o1.getScores().get(0));
            }
        });
        return temp_dist_pair;
    }


    public Integer getMesPow(String uid){
        /* Questa funzione viene utilizzata per ottenere il Measured Power del beacon
        andando a visitare il dizionario in cui queste sono memorizzate
        **/
        Map<String, Integer> dict_mes_pow = new HashMap<String, Integer>(){
            {
                put("D8:59:D1:1A:53:81", -60); //mi_band senza scotch
                put("3C:A3:08:AE:1E:C0", -70);
                put("3C:A3:08:AE:24:01", -80);
                put("3C:A3:08:AE:24:02", -90);
                put("3C:A3:08:AE:18:F2", -100);
                put("DB:9D:52:29:F4:84", -65); //mi band con scotch

            }
        };
        return dict_mes_pow.get(uid);
    }

    public Coord get_position(List<DistPair> l){
        /* Questa funzione utilizza il dizionario map_coord_beacon che contiene le coordinate di ogni beacon.
        Restituisce un oggetto Coord che verrà inserito in una lista, in modo da avere un insieme di coordinate.
        All'interno di questa funzione viene applicato il calcolo della distanza usando il Friis free space model;
        il risultato viene inserito all'interno di list_position.
        Inizialmente bisogna andare a scegliere gli elementi che hanno un rssi che più si avvicina allo zero, quindi
        bisogna ordinare la lista e prendere i primi tre.
        */
        //Ottenere i primi tre oggetti della lista
        //map_coord_beacon.get(l.get(0).device).x;
        Float x1 = (float) map_coord_beacon.get(l.get(0).device).x;
        Float y1 = (float) map_coord_beacon.get(l.get(0).device).y;
        Float r1 = (float)Math.pow(l.get(0).dist,2);

        Float x2 = (float) map_coord_beacon.get(l.get(1).device).x;
        Float y2 = (float) map_coord_beacon.get(l.get(1).device).y;
        Float r2 = (float)Math.pow(l.get(1).dist,2);

        Float x3 = (float) map_coord_beacon.get(l.get(2).device).x;
        Float y3 = (float) map_coord_beacon.get(l.get(2).device).y;
        Float r3 = (float)Math.pow(l.get(2).dist,2);

        Float a = (2*x2) - (2*x1);
        Float b = 2*y2 - 2*y1;
        Float c = (float) (Math.pow(r1,2) - Math.pow(r2,2) - Math.pow(x1,2) + Math.pow(x2,2) - Math.pow(y1,2) + Math.pow(y2,2));
        Float d = 2*x3 - 2*x2;
        Float e = 2*y3 - 2*y2;
        Float f = (float) (Math.pow(r2,2) - Math.pow(r3,2) - Math.pow(x2,2) + Math.pow(x3,2) - Math.pow(y2,2) + Math.pow(y3,2));
        Float x = (c*e - f*b)/(e*a - b*d);
        Float y = (c*d - a*f)/(b*d - a*e);

        return new Coord(Math.round(x), Math.round(y));
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnScan = (Button) findViewById(R.id.btnScan);
        textv = findViewById(R.id.s7);
        textv_coordinate = findViewById(R.id.coordinate);
        BluetoothManager btManager = (BluetoothManager)getSystemService(Context.BLUETOOTH_SERVICE);
        this.mBluetoothAdapter = btManager.getAdapter();
        this.mBluetoothLeScanner = mBluetoothAdapter.getBluetoothLeScanner();
        this.mHandler = new Handler();

        enableBt();
        checkBtPermissions();

    }

    public void onBtnScan(View v) {
        Log.v("BLE", "dentro onBtnScan");
        if (mScanning) {
            mScanning = false;
            scanLeDevice(false);
            btnScan.setText("START SCAN");
            Log.v("BLE", "Start scan fermo");

        } else {
            mScanning = true;
            scanLeDevice(true);
            Log.v("BLE", "start scan avviato");

            btnScan.setText("STOP SCAN");
        }
    }

    public void checkBtPermissions() {
        this.requestPermissions(
                new String[]{
                        Manifest.permission.ACCESS_COARSE_LOCATION,
                        Manifest.permission.BLUETOOTH_CONNECT, BLUETOOTH_SCAN,
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.BLUETOOTH
                },
                REQUEST_BT_PERMISSIONS);
    }

    public void enableBt() {
        if (mBluetoothAdapter == null) {
            // Device does not support Bluetooth
            Log.v("BLE", "Device non supporta Blututto");
        }
        if (!mBluetoothAdapter.isEnabled()) {
            Intent enableBtIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
                // TODO: Consider calling
                //    ActivityCompat#requestPermissions
                // here to request the missing permissions, and then overriding
                //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                //                                          int[] grantResults)
                // to handle the case where the user grants the permission. See the documentation
                // for ActivityCompat#requestPermissions for more details.
                Log.v("BLE", "error permission enableBt");
                checkBtPermissions();
                return;
            }
            startActivityForResult(enableBtIntent, REQUEST_BT_ENABLE);
        }
    }

    public void scanLeDevice(final boolean enable) {
        if (enable) {
            mScanning = true;
            Log.v("BLE", "start");

            Log.v("BLE","return "+ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION));
            Log.v("BLE", "return " + ActivityCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN));
            Log.v("BLE","expect "+PackageManager.PERMISSION_GRANTED);
            checkBtPermissions();
            boolean perm_android12 = ActivityCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED;
            boolean perm_android11 = ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED;

            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.R){
                if(perm_android12){
                    Log.v("BLE", "errore in bluetooth_scan");
                    checkBtPermissions();
                    return;
                }
            }
            else if(Build.VERSION.SDK_INT <= Build.VERSION_CODES.R){
                if(perm_android11){
                    Log.v("BLE", "errore in bluetooth_scan");
                    checkBtPermissions();
                    return;
                }
            }

            Log.v("BLE", "avvio il bluetooth_scan");
            mBluetoothLeScanner.startScan(mLeScanCallback);
            startRepeatingTask();
        } else {
            Log.v("BLE", "stop");
            mScanning = false;
            mBluetoothLeScanner.stopScan(mLeScanCallback);
            stopRepeatingTask();
        }
    }
}

