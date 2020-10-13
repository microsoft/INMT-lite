package com.example.inmt_offline.UI;

import android.content.Intent;
import android.util.Pair;
import android.view.View;

import java.util.ArrayList;

public class Preview_Button {
    public static void onClick(View view, ArrayList<Pair<String, String>> sentencePairs, Temp temp) {
        /*
        * sentencePair is a ArrayList of pairs where first element is the source sentence
        * the second element is the translated sentence
        * WRITE YOUR CUSTOM OWN CODE BELOW
        *
        * */


        /*
        * BELOW CODE IS AN EXAMPLE SHOWING WORKING WITH TRANSLATION PAIRS
        * REMOVE BELOW CODE IN CASE PROVIDING YOUR OWN IMPLEMENTATION
        * */

        Intent intent = new Intent(temp.getApplicationContext(), PreviewActivity.class);

        ArrayList<String> qTrans = new ArrayList<>();
        ArrayList<String> aTrans = new ArrayList<>();

        for(Pair p: sentencePairs) {
            qTrans.add((String) p.first);
            aTrans.add((String) p.second);
        }

        intent.putExtra("qTrans", qTrans);
        intent.putExtra("aTrans", aTrans);

        temp.startActivity(intent);

    }
}
