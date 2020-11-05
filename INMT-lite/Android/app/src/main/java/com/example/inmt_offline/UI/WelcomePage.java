package com.example.inmt_offline.UI;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.inmt_offline.R;

public class WelcomePage extends AppCompatActivity {

    Spinner opModeSpinner;
    Spinner transTypeSpinner;
    Button loadSampleButton;
    Button translateButton;
    EditText sourceEditText;

    static public int OFFLINE_MODE = 0;
    static public int ONLINE_MODE = 1;

    public static final int ENGLISH_HINDI_ONLINE = 0;
    public static final int HINDI_ENGLISH_ONLINE = 1;
    public static final int HINDI_GONDI_ONLINE = 2;
    public static final int CUSTOM_ONLINE = 3;


    public static final int HINDI_GONDI_OFFLINE = 0;
    public static final int CUSTOM_OFFLINE = 1;

    int mode = 0; // Corresponds to the type of mode, 0: OFFLINE, 1: ONLINE
    int lang_spec = 0; // Corresponds to the type translation pair, 0: English-Hindi, 1: Hindi-English, 2: Hindi-Gondi, 3: Custom

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_welcome_page);

        getSupportActionBar().hide();

        opModeSpinner = findViewById(R.id.opModeSpinner);
        transTypeSpinner = findViewById(R.id.transTypeSpinner);
        loadSampleButton = findViewById(R.id.loadSampleButton);
        translateButton = findViewById(R.id.translateButton);
        sourceEditText = findViewById(R.id.sourceEditText);

        final String Modes[] = getResources().getStringArray(R.array.Modes);
        String offlineTransOPs[] = getResources().getStringArray(R.array.offlineTransOPs);
        String onlineTransOPs[] = getResources().getStringArray(R.array.onlineTransOPs);

        ArrayAdapter<String> opAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_dropdown_item_1line, Modes);
        final ArrayAdapter<String> offlineAdapter = new ArrayAdapter<String>(getApplicationContext(), android.R.layout.simple_dropdown_item_1line, offlineTransOPs);
        final ArrayAdapter<String> onlineAdapter = new ArrayAdapter<String>(getApplicationContext(), android.R.layout.simple_dropdown_item_1line, onlineTransOPs);

        opAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        opModeSpinner.setAdapter(opAdapter);

        opModeSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                if (i == 0) {
                    // Set Adapter for offline mode
                    transTypeSpinner.setAdapter(offlineAdapter);
                } else {
                    // Set Adapter for online mode
                    transTypeSpinner.setAdapter(onlineAdapter);
                }
                mode = i;
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });

        transTypeSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                lang_spec = i;
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });


        // Translate Button Click Listener
        translateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (sourceEditText.getText().toString().isEmpty()) {
                    Toast.makeText(WelcomePage.this, "Please enter a sentence to translate.", Toast.LENGTH_SHORT).show();
                } else {
                    splitAndStartActivity();
                }

            }
        });

        loadSampleButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if (mode == ONLINE_MODE) {

                    switch (lang_spec) {
                        case ENGLISH_HINDI_ONLINE:
                            sourceEditText.setText(R.string.sampleEnglishString);
                            break;
                        case HINDI_ENGLISH_ONLINE:
                            sourceEditText.setText(R.string.sampleHindiString);
                            break;
                        case HINDI_GONDI_ONLINE:
                            sourceEditText.setText(R.string.sampleHindiString);
                            break;
                        case CUSTOM_ONLINE:
                            sourceEditText.setText(R.string.CustomLoadSample_ONLINE);
                            break;

                    }

                } else {

                    switch (lang_spec) {
                        case HINDI_GONDI_OFFLINE:
                            sourceEditText.setText(R.string.sampleHindiString);
                            break;
                        case CUSTOM_OFFLINE:
                            sourceEditText.setText(R.string.CustomLoadSample_OFFLINE);

                    }

                }



            }
        });


    }

    void splitAndStartActivity() {
        String[] qTransStrings = sourceEditText.getText().toString().toLowerCase().split("[!?।|.](?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
        Intent intent = new Intent(getApplicationContext(), Temp.class);
        intent.putExtra("qTransStrings", qTransStrings);
        intent.putExtra("mode", mode);
        intent.putExtra("lang_spec", lang_spec);
        startActivity(intent);
    }
}