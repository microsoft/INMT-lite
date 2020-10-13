package com.example.inmt_offline.UI;

import android.content.Intent;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.inmt_offline.Adapters.PreviewListAdapter;
import com.example.inmt_offline.R;

import java.util.ArrayList;

public class PreviewActivity extends AppCompatActivity {

    RecyclerView previewRecyclerView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_preview);

        getSupportActionBar().setTitle("Preview");
        getSupportActionBar().setDisplayShowHomeEnabled(true);

        previewRecyclerView = findViewById(R.id.previewRecyclerView);
        Intent intent = getIntent();
        ArrayList<String> qTrans = intent.getStringArrayListExtra("qTrans");
        ArrayList<String> aTrans = intent.getStringArrayListExtra("aTrans");

        LinearLayoutManager layoutManager = new LinearLayoutManager(this);
        previewRecyclerView.setLayoutManager(layoutManager);
        PreviewListAdapter adapter = new PreviewListAdapter(qTrans, aTrans);

        previewRecyclerView.setAdapter(adapter);

    }
}