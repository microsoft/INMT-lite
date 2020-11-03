package com.example.inmt_offline.Adapters;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.res.AssetFileDescriptor;
import android.os.Handler;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.recyclerview.widget.RecyclerView;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.example.inmt_offline.R;
import com.example.inmt_offline.UI.Temp;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

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

import static com.example.inmt_offline.UI.WelcomePage.OFFLINE_MODE;


public class TransListAdapter extends RecyclerView.Adapter<TransListAdapter.MyViewHolder> {
    private String[] qTransStrings;
    private String[] lang_spec_codes_online;
    RequestQueue requestQueue;
    Temp translationActivityThis;
    int mode;
    int lang_spec;
    JSONObject inp_tokenizer_json;
    JSONObject tgt_tokenizer_json;
    private Interpreter tfLite;
    private String end_token = "<end>";
    private ArrayList<String> inp_tokenizer = new ArrayList<>();
    private ArrayList<String> tgt_tokenizer = new ArrayList<>();
    boolean focus[];
    String BASE_URL;
    long delay = 500; // 1 seconds after user stops typing
    long last_text_edit = 0;
    Handler handler;

    public static class MyViewHolder extends RecyclerView.ViewHolder {
        // each data item is just a string in this case
        public TextView qTransTextView;
        public AutoCompleteTextView aTransEditText;

        public MyViewHolder(View v) {
            super(v);
            qTransTextView = v.findViewById(R.id.qTransTextView);
            aTransEditText = v.findViewById(R.id.aTransEditText);
        }
    }

    private int getTokenNumber(ArrayList<String> list, String key) {

        for (int i = 0; i < list.size(); i++) {
            if (Normalizer.normalize(list.get(i), Normalizer.Form.NFD).equals(Normalizer.normalize(key, Normalizer.Form.NFD)))
                return i + 1;
        }

        return 0;
    }

    public String loadJSONFromAsset(String name) {
        String json = null;
        try {
            InputStream is = translationActivityThis.getAssets().open(name);
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

    private void loadJson() {

        ProgressDialog pd = new ProgressDialog(translationActivityThis);

        pd.setMessage("Loading Data..");

        pd.show();

        try {
            inp_tokenizer_json = new JSONObject(loadJSONFromAsset(translationActivityThis.getString(R.string.INPUT_JSON_VOCAB)));
            tgt_tokenizer_json = new JSONObject(loadJSONFromAsset(translationActivityThis.getString(R.string.TARGET_JSON_VOCAB)));
            for (int i = 1; i < tgt_tokenizer_json.length(); i++)
                tgt_tokenizer.add((String) tgt_tokenizer_json.get(String.valueOf(i)));

            for (int i = 1; i < inp_tokenizer_json.length(); i++)
                inp_tokenizer.add((String) inp_tokenizer_json.get(String.valueOf(i)));
        } catch (JSONException e) {
            e.printStackTrace();
        }

        pd.dismiss();

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
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        Log.i("Model read:", "started");
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(translationActivityThis.getString(R.string.MODEL_FILE));
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        Log.i("Model read:", "success");
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Provide a suitable constructor (depends on the kind of dataset)
    public TransListAdapter(String[] qTransStrings, Temp translationActivityThis, int mode, int lang_spec) {
        this.qTransStrings = qTransStrings;
        this.translationActivityThis = translationActivityThis;
        this.lang_spec = lang_spec;
        this.mode = mode;
        BASE_URL = translationActivityThis.getResources().getString(R.string.BASE_URL);
        lang_spec_codes_online = translationActivityThis.getResources().getStringArray(R.array.lang_spec_codes_online);
        handler = new Handler();
        if (mode == 1) {
            requestQueue = Volley.newRequestQueue(translationActivityThis);
        }

        try {
            tfLite = new Interpreter(loadModelFile(translationActivityThis));
        } catch (IOException e) {
            Toast.makeText(translationActivityThis, "", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
        focus = new boolean[qTransStrings.length];
        loadJson();

    }

    // Create new views (invoked by the layout manager)
    @Override
    public TransListAdapter.MyViewHolder onCreateViewHolder(ViewGroup parent,
                                                            int viewType) {
        // create a new view
        View v = LayoutInflater.from(translationActivityThis)
                .inflate(R.layout.translation_card, parent, false);

        MyViewHolder vh = new MyViewHolder(v);
        return vh;
    }


    // Replace the contents of a view (invoked by the layout manager)
    @Override
    public void onBindViewHolder(final MyViewHolder holder, final int position) {
        // - get element from your dataset at this position
        // - replace the contents of the view with that element


        holder.qTransTextView.setText(qTransStrings[position]);
        final float[][] mask = new float[1][tgt_tokenizer.size() + 2];

        for (int i = 0; i < qTransStrings.length; i++) focus[i] = false;

        final Runnable input_finish_checker = new Runnable() {
            public void run() {
                if (System.currentTimeMillis() > (last_text_edit + delay - 500)) {
                    // TODO: do what you need here
                    // ............
                    // ............
                    if (focus[position]) {
                        String inp_string = holder.qTransTextView.getText().toString();

                        String tgt_string = holder.aTransEditText.getText().toString();
                        if (mode == OFFLINE_MODE) {
                            try {
                                int start_index = 2;
                                int end_index = 3;
                                String splitted[] = tgt_string.split(" ");


                                Boolean applyMask = false;// at model's index 2

                                for (int i = 0; i < mask[0].length; i++) mask[0][i] = 1;


                                Log.i("Len vocab:", tgt_tokenizer.size() + "");

                                if (tgt_string.length() > 1)
                                    if (tgt_string.charAt(tgt_string.length() - 1) != ' ') {
                                        applyMask = true;
                                    }
                                // Splitting the input string by spaces
                                final String[] tgt_words = !tgt_string.equals("") ? tgt_string.split(" ") : " ".split(" ");
                                // Splitting the user input by spaces
                                String[] inp_words = !inp_string.equals("") ? inp_string.split(" ") : " ".split(" ");


                                Log.i("inp_vector_len", inp_words.length + "");

                                for (String i : inp_words) {
                                    Log.i("inp_str", i);
                                }

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
                                while (w_i >= 0 && index >= 0) {
                                    tgt_inputArray[0][index] = getTokenNumber(tgt_tokenizer, tgt_words[w_i]);
                                    w_i--;
                                    index--;
                                }
                                if (index >= 0) tgt_inputArray[0][index] = start_index;

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
//                        }


                                final ArrayList<String> res = runModel(inp_inputArray, tgt_inputArray, mask);

                                for (String sugg : res) {
                                    if (sugg.equals(end_token)) {
                                        throw new Exception("End Of Translation");
                                    }
                                }


                                final String curr = tgt_string;
                                final Boolean finalApplyMask = applyMask;
                                holder.aTransEditText.setOnItemClickListener(new AdapterView.OnItemClickListener() {
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
                                        to_set += res.get(position) + " ";
                                        holder.aTransEditText.setText(to_set);
                                        holder.aTransEditText.setSelection(to_set.length());
                                    }
                                });
                                ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(holder.itemView.getContext(), android.R.layout.simple_list_item_1, res);
                                holder.aTransEditText.setAdapter(arrayAdapter);
                                holder.aTransEditText.showDropDown();

                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        } else {

                            String GET_URL = BASE_URL + "api/simple/translate_new?langspec=" + lang_spec_codes_online[lang_spec] + "&sentence=" + URLEncoder.encode(inp_string);

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

                                        for (int i = 0; i < suggestionJSONArray.length(); i++) {
                                            suggestionArray[i] = suggestionJSONArray.getString(i);
                                        }


                                        holder.aTransEditText.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                                            @Override
                                            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                                                String to_set = holder.aTransEditText.getText().toString();
                                                to_set = suggestionArray[position];
                                                holder.aTransEditText.setText(to_set);
                                                holder.aTransEditText.setSelection(to_set.length());
                                            }
                                        });

                                        ArrayAdapter<String> arrayAdapter = new ArrayAdapter<>(translationActivityThis, android.R.layout.simple_list_item_1, suggestionArray);
                                        holder.aTransEditText.setAdapter(arrayAdapter);
                                        holder.aTransEditText.showDropDown();
                                    } catch (JSONException e) {
                                        Toast.makeText(translationActivityThis, e.toString(), Toast.LENGTH_SHORT).show();
                                        e.printStackTrace();
                                    }
                                }
                            }, new Response.ErrorListener() {
                                @Override
                                public void onErrorResponse(VolleyError error) {
                                    error.printStackTrace();
                                    Toast.makeText(translationActivityThis, error.toString(), Toast.LENGTH_SHORT).show();
                                }
                            });

                            requestQueue.add(suggestionReq);


                        }
                    }
                }
            }
        };

        holder.aTransEditText.setOnFocusChangeListener(new View.OnFocusChangeListener() {
            @Override
            public void onFocusChange(View view, boolean b) {
                focus[position] = b;
                last_text_edit = System.currentTimeMillis();
                handler.postDelayed(input_finish_checker, -1000);
            }
        });

        holder.aTransEditText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void onTextChanged(CharSequence charSequence, int i1, int i2, int i3) {
                handler.removeCallbacks(input_finish_checker);
            }

            @Override
            public void afterTextChanged(Editable s) {

                last_text_edit = System.currentTimeMillis();
                handler.postDelayed(input_finish_checker, delay);

            }
        });




    }

    // Return the size of your dataset (invoked by the layout manager)
    @Override
    public int getItemCount() {
        return qTransStrings.length;
    }
}