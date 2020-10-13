package com.example.inmt_offline.UI;


import android.content.Intent;
import android.os.Bundle;
import android.util.Pair;
import android.view.View;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.inmt_offline.Adapters.TransListAdapter;
import com.example.inmt_offline.External.Preview_Button;
import com.example.inmt_offline.R;

import java.util.ArrayList;

import static com.example.inmt_offline.UI.WelcomePage.OFFLINE_MODE;

public class Temp extends AppCompatActivity {

    RecyclerView transRecyclerView;
    TextView previewTextView;
    TextView transTitleTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_translation);

        transRecyclerView = findViewById(R.id.transRecyclerView);
        previewTextView = findViewById(R.id.previewTextView);
        transTitleTextView = findViewById(R.id.transTitleTextView);
        String offlineTransOPs[] = getResources().getStringArray(R.array.offlineTransOPs);
        String onlineTransOPs[] = getResources().getStringArray(R.array.onlineTransOPs);

        Intent intent = getIntent();
        final String[] qTransStrings = intent.getStringArrayExtra("qTransStrings");
        int mode = intent.getIntExtra("mode", -1);
        int lang_spec = intent.getIntExtra("lang_spec", -1);

        if (mode == OFFLINE_MODE) {
            transTitleTextView.setText(offlineTransOPs[lang_spec]);
        } else {
            transTitleTextView.setText(onlineTransOPs[lang_spec]);
        }


        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        transRecyclerView.setLayoutManager(linearLayoutManager);

        TransListAdapter transListAdapter = new TransListAdapter(qTransStrings, this, mode, lang_spec);

        transRecyclerView.setAdapter(transListAdapter);

        previewTextView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                ArrayList<Pair<String, String>> sentencePair = new ArrayList<>();

                for(int i=0; i<transRecyclerView.getChildCount(); i++) {
                    TransListAdapter.MyViewHolder holder = (TransListAdapter.MyViewHolder) transRecyclerView.getChildViewHolder(transRecyclerView.getChildAt(i));
                    sentencePair.add(new Pair<String, String>(qTransStrings[i], holder.aTransEditText.getText().toString()));
                }

                Preview_Button.onClick(view, sentencePair, Temp.this);
            }
        });


    }
}
