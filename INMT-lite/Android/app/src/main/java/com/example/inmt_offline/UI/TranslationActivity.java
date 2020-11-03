package com.example.inmt_offline.UI;


import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.example.inmt_offline.R;
import com.jakewharton.rxbinding.widget.RxTextView;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URLEncoder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

import rx.android.schedulers.AndroidSchedulers;
import rx.functions.Action1;

public class TranslationActivity extends AppCompatActivity {


    private Interpreter tfLite;

    private ArrayList<String> inp_tokenizer = new ArrayList<>();
    private ArrayList<String> tgt_tokenizer = new ArrayList<>();

    private Button translateButton;
    private Switch modeSwitch;
    private EditText inp_userInput;
    private AutoCompleteTextView tgt_userInput;
    private TextView outputSentence;
    private String BASE_URL;
    private TextView modeTextView;
    Boolean offline_mode = true;

    JSONObject inp_tokenizer_json;
    JSONObject tgt_tokenizer_json;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        translateButton = findViewById(R.id.button2);
        inp_userInput = findViewById(R.id.inp_EditText);
        tgt_userInput = findViewById(R.id.tgt_EditText);
        outputSentence = findViewById(R.id.textView);
        modeSwitch = findViewById(R.id.modeSwitch);
        modeTextView = findViewById(R.id.modeTextView);

        BASE_URL = getString(R.string.INMT_URL);

        final ArrayList<String>[] res = new ArrayList[1];

        loadJson();

        final float[][] mask = new float[1][tgt_tokenizer.size() + 2];

        modeSwitch.setChecked(true);
        final RequestQueue requestQueue = Volley.newRequestQueue(getApplicationContext());

        modeSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                offline_mode = isChecked;
                Log.i("OFFLINE_MODE_IS", offline_mode + " " + isChecked);
                if (isChecked) modeTextView.setText("OFFLINE MODE");
                else modeTextView.setText("ONLINE MODE");
            }
        });


        RxTextView.textChanges(tgt_userInput)
                .debounce(100, TimeUnit.MILLISECONDS)
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(new Action1<CharSequence>() {
                    @Override
                    public void call(CharSequence textChanged) {
                        Log.i("RxaTransEditText", "Stopped Typing");

                        String inp_string = inp_userInput.getText().toString();
                        String tgt_string = tgt_userInput.getText().toString();

                        Log.i("OFFLINE_MODE", offline_mode+"");

                        if (offline_mode) {
                            try {

                                Boolean applyMask = false;// at model's index 2

                                for (int i = 0; i < mask[0].length; i++) mask[0][i] = 1;


                                Log.i("Len vocab:", tgt_tokenizer.size() + "");

                                if (tgt_string.length() > 1)
                                    if (tgt_string.charAt(tgt_string.length() - 1) != ' ') {
                                        applyMask = true;
                                    }

                                final String[] tgt_words = tgt_string.split(" ");
                                String[] inp_words = inp_string.split(" ");

                                int inp_wordsLength = inp_words.length;
                                int tgt_wordsLength = tgt_words.length;

//                        float[][] inp_inputArray = getArray(inp_words, inp_wordsLength, inp_tokenizer);

                                int Tx = 14;
                                float[][] inp_inputArray = new float[1][Tx];

                                inp_inputArray[0][0] = 2;
                                int mini = inp_wordsLength;
                                if (mini > Tx - 1) {
                                    mini = Tx - 1;
                                }
                                for (int i = 1; i <= mini; i++) {
                                    inp_inputArray[0][i] = getTokenNumber(inp_tokenizer, inp_words[i - 1]);
                                }

                                if (mini + 1 < Tx) {
                                    inp_inputArray[0][mini + 1] = 3;
                                }


                                float[][] tgt_inputArray = new float[1][Tx];
//                        tgt_inputArray = getArray(tgt_words, tgt_wordsLength, tgt_tokenizer);

                                int w_i = tgt_wordsLength - 1;
                                if (applyMask)
                                    w_i -= 1; // Do not consider last word when mask is applied
                                int index = Tx - 1;
                                while (w_i >= 0 && index >= 1) {
                                    tgt_inputArray[0][index] = getTokenNumber(tgt_tokenizer, tgt_words[w_i]);
                                    w_i--;
                                    index--;
                                }

                                Log.i("APPLY_MASK_VALUE", applyMask + "");

                                if (applyMask) {
                                    for (int i = 0; i < tgt_tokenizer.size(); i++) {
                                        Boolean check = Normalizer.normalize(tgt_tokenizer.get(i), Normalizer.Form.NFD).startsWith(Normalizer.normalize(tgt_words[tgt_words.length - 1], Normalizer.Form.NFD));
                                        if (!check) {
                                            mask[0][i + 1] = 0;
                                        } else {
                                            Log.i("starts with ::", tgt_words[tgt_words.length - 1] + " " + check + " " + tgt_tokenizer.get(i));
                                        }
                                    }
                                }

                                Log.i("tgt_words_len", tgt_words.length + "");

                                for (int i = 0; i < tgt_words.length; i++)
                                    Log.i("tgt_words", tgt_words[i]);

                                if (applyMask) {
                                    Log.i("mask_bool check", applyMask + "");
                                    Log.i("whole_mask", Arrays.toString(mask[0]));
                                }


//                        int dist = 0;
//                        for(float i: inp_inputArray[0]) {
//                            if (i == 0) dist++;
//                            else break;
//                        }
//                        for (int i=0; i < inp_inputArray[0].length; i++){
//                            if (i+dist < inp_inputArray[0].length) {
//                                inp_inputArray[0][i] = inp_inputArray[0][i+dist];
//                            } else {
//                                inp_inputArray[0][i] = 0;
//                            }
//                        }


                                res[0] = runModel(inp_inputArray, tgt_inputArray, mask);
                                final String curr = tgt_userInput.getText().toString();
                                ArrayAdapter<String> arrayAdapter = new ArrayAdapter<>(getApplicationContext(), android.R.layout.simple_list_item_1, res[0]);

                                final Boolean finalApplyMask = applyMask;
                                tgt_userInput.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                                    @Override
                                    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                                        Log.i("tgt_words_listner", Arrays.toString(tgt_words));
                                        String to_set = "";
                                        if (finalApplyMask) {
                                            for (int i = 0; i < tgt_words.length - 1; i++) {
                                                to_set += tgt_words[i] + " ";
                                            }
                                        } else {
                                            for (String s : tgt_words) {
                                                to_set += s + " ";
                                            }
                                        }
                                        to_set += res[0].get(position) + " ";
                                        tgt_userInput.setText(to_set);
                                        tgt_userInput.setSelection(to_set.length());
                                    }
                                });


                                tgt_userInput.setAdapter(arrayAdapter);


                                tgt_userInput.showDropDown();
                            } catch (Exception e) {
                                e.printStackTrace();
                            }

                        }
                        else {

                            String GET_URL = BASE_URL + "api/simple/translate_new?langspec=" + "en-hi" + "&sentence=" + URLEncoder.encode(inp_string);

                            Log.i("GET_URL", GET_URL);

                            if (tgt_string.length() > 0) {
                                GET_URL += "&partial_trans=" + tgt_string;
                            }

                            StringRequest suggestionReq = new StringRequest(Request.Method.GET, GET_URL, new Response.Listener<String>() {
                                @Override
                                public void onResponse(String response) {
                                    try {
                                        Log.i("response", response);
                                        JSONObject JSONresponse = new JSONObject(response);
                                        final JSONArray suggestionJSONArray = JSONresponse.getJSONArray("result");
                                        JSONArray attentionArray = JSONresponse.getJSONArray("attn");
                                        double avg = JSONresponse.getDouble("avg");
                                        String partial = JSONresponse.getString("partial");
                                        double ppl = JSONresponse.getDouble("ppl");

                                        final String[] suggestionArray = new String[suggestionJSONArray.length()];
//                                    String[] suggestionArray = {"Anurag", "Shukla"};
//                                    ArrayAdapter<String> arrayAdapter = new ArrayAdapter<>(translationActivityThis, android.R.layout.simple_list_item_1, suggestionArray);
//                                    holder.aTransEditText.setAdapter(arrayAdapter);

                                        for(int i=0; i<suggestionJSONArray.length(); i++) {
                                            suggestionArray[i] = suggestionJSONArray.getString(i);
                                        }


                                        tgt_userInput.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                                            @Override
                                            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                                                String to_set = tgt_userInput.getText().toString();
                                                to_set = suggestionArray[position];
                                                tgt_userInput.setText(to_set);
                                                tgt_userInput.setSelection(to_set.length());
                                            }
                                        });

                                        ArrayAdapter<String> arrayAdapter = new ArrayAdapter<>(getApplicationContext(), android.R.layout.simple_list_item_1, suggestionArray);
                                        tgt_userInput.setAdapter(arrayAdapter);
                                        tgt_userInput.showDropDown();
                                    } catch (JSONException e) {
                                        Toast.makeText(getApplicationContext(), e.toString(), Toast.LENGTH_SHORT).show();
                                        e.printStackTrace();
                                    }
                                }
                            }, new Response.ErrorListener() {
                                @Override
                                public void onErrorResponse(VolleyError error) {
                                    error.printStackTrace();
                                    Toast.makeText(getApplicationContext(), error.toString(), Toast.LENGTH_SHORT).show();
                                }
                            });

                            requestQueue.add(suggestionReq);


                        }
                    }
                });


        try {
            tfLite = new Interpreter(loadModelFile(this));
        } catch (IOException e) {
            e.printStackTrace();
        }


//        translateButton.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//
//                String[] inp_words = inp_userInput.getText().toString().split("\\s+");
//                String[] tgt_words = tgt_userInput.getText().toString().split("\\s+");
//
//                int inp_wordsLength = inp_words.length;
//                int tgt_wordsLength = tgt_words.length;
//
//                float[][] inp_inputArray = getArray(inp_words, inp_wordsLength, inp_tokenizer);
//                float[][] tgt_inputArray = getArray(tgt_words, tgt_wordsLength, tgt_tokenizer);
//
//                int dist = 0;
//                for(float i: inp_inputArray[0]) {
//                    if (i == 0) dist++;
//                    else break;
//                }
//                for (int i=0; i < inp_inputArray[0].length; i++){
//                    if (i+dist < inp_inputArray[0].length) {
//                        inp_inputArray[0][i] = inp_inputArray[0][i+dist];
//                    } else {
//                        inp_inputArray[0][i] = 0;
//                    }
//                }
//
//
//
//
//                String res = runModel(inp_inputArray, tgt_inputArray);
//                outputSentence.setText(res);
//            }
//        });


    }

    private float[][] getArray(String[] words, int wordsLength, ArrayList<String> tokenizer) {

        int Tx = 14;
        float[][] arr = new float[1][Tx];

        int k = 0;
        if (wordsLength > Tx - 1) {
            arr[0][0] = 2;
            for (int i = 1; i < Tx; i++)
                arr[0][i] = getTokenNumber(tokenizer, words[i - 1]);
        } else {
            for (int i = 1; i < Tx; i++) {
                if (i < Tx - 1 - wordsLength)
                    arr[0][i] = 0.0f;
                else if (i == Tx - 1 - wordsLength) arr[0][i] = 2;
                else {
                    arr[0][i] = getTokenNumber(tokenizer, words[k]);
                    Log.i("INPUT " + words[k], " " + arr[0][i]);
                    k++;

                }
            }
        }

        return arr;


    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        Log.i("Model read:", "started");
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getString(R.string.MODEL_FILE));
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        Log.i("Model read:", "success");
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    private void loadJson() {

        ProgressDialog pd = new ProgressDialog(TranslationActivity.this);

        pd.setMessage("Loading Data..");

        pd.show();

        try {
            inp_tokenizer_json = new JSONObject(loadJSONFromAsset(getString(R.string.INPUT_JSON_VOCAB)));
            tgt_tokenizer_json = new JSONObject(loadJSONFromAsset(getString(R.string.TARGET_JSON_VOCAB)));
            for (int i = 1; i < tgt_tokenizer_json.length(); i++)
                tgt_tokenizer.add((String) tgt_tokenizer_json.get(String.valueOf(i)));

            for (int i = 1; i < inp_tokenizer_json.length(); i++)
                inp_tokenizer.add((String) inp_tokenizer_json.get(String.valueOf(i)));
        } catch (JSONException e) {
            e.printStackTrace();
        }

        pd.dismiss();

    }


    public String loadJSONFromAsset(String name) {
        String json = null;
        try {
            InputStream is = TranslationActivity.this.getAssets().open(name);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            json = new String(buffer, "UTF-8");
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
        return json;
    }


    private int getTokenNumber(ArrayList<String> list, String key) {

        for (int i = 0; i < list.size(); i++) {
            if (Normalizer.normalize(list.get(i), Normalizer.Form.NFD).equals(Normalizer.normalize(key, Normalizer.Form.NFD)))
                return i + 1;
        }

        return 0;
    }


    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    private ArrayList<String> runModel(float[][] inp_inputVal, float[][] tgt_inputVal, float[][] mask) {


        Log.i("Model Run: ", "started");
        HashMap<Integer, Object> outputVal = new HashMap<>();
        outputVal.put(0, new float[1][5]);
        outputVal.put(1, new int[1][5]);
        float[][] enc_hidden = new float[1][1024];
        float[][] dec_input = new float[1][1];

        dec_input[0][0] = 2;

        String inp_arr = "";
        for (float i : inp_inputVal[0]) inp_arr += i + " ";
        Log.i("inp_inputVal", inp_arr);

        String tgt_arr = "";
        for (float i : tgt_inputVal[0]) tgt_arr += i + " ";
        Log.i("tgt_inputVal", tgt_arr);

        tfLite.runForMultipleInputsOutputs(new Object[]{dec_input, enc_hidden, mask, tgt_inputVal, inp_inputVal}, outputVal);

        Log.i("MODEL: ", "run success");

        int[][] top5 = (int[][]) outputVal.get(1);

        ArrayList<String> res = new ArrayList<>();

        for (int i : top5[0]) {
            if (i == 0) res.add("pad");
            else {
                res.add(tgt_tokenizer.get(i - 1));
                Log.i("TOP 5", tgt_tokenizer.get(i - 1) + " " + (i - 1));
            }
        }

        return res;
//        StringBuilder stringBuilder = new StringBuilder();

//        for(int i=0; i< 8; i++) {
//            float[][] floats = (float[][]) outputVal.get(i);
//            stringBuilder.append(getWordFromToken(list,argMax(floats[0])));
//            String str = "";
//            for(int j=0; j<8; j++) {
//
//                str += inputVal[0][j]  + "----";
//            }
//
//            Log.i("VAL " + i, str);
//
//            stringBuilder.append(" ");
//        }

//        return stringBuilder.toString();
    }

    private String getWordFromToken(ArrayList<String> list, int key) {

        if (key == 0)
            return "";
        else
            return list.get(key - 1);

    }

    private static int argMax(float[] floatArray) {

        float max = floatArray[0];
        int index = 0;

        for (int i = 0; i < floatArray.length; i++) {
            if (max < floatArray[i]) {
                max = floatArray[i];
                index = i;
            }
        }
        return index;
    }


}