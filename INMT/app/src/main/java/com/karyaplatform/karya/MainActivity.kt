package com.karyaplatform.karya

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import com.karyaplatform.karya.inmt.enum.MODE

class MainActivity : AppCompatActivity() {
    private lateinit var dropdownBtn: Button
    private lateinit var bowBtn: Button
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        dropdownBtn = findViewById(R.id.dropdownBtn)
        bowBtn = findViewById(R.id.bowBtn)

        dropdownBtn.setOnClickListener { navigateToTranslationActivity(MODE.DROPDOWN) }
        bowBtn.setOnClickListener { navigateToTranslationActivity(MODE.BOW) }
    }

    private fun navigateToTranslationActivity(mode: MODE) {
        val intent = Intent(this, TranslationActivity::class.java)
        intent.putExtra("mode", mode.toString())
        startActivity(intent)
    }
}