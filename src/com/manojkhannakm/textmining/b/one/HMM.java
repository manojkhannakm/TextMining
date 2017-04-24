package com.manojkhannakm.textmining.b.one;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

import java.io.*;
import java.util.*;

/**
 * Hidden Markov Model
 *
 * @author Manoj Khanna
 */

public class HMM {

    private static StanfordCoreNLP pipeline;
    private static Trie stopWordTrie;

    public static void main(String[] args) throws IOException {
        System.out.println("Reading input.txt...");
        long startTime = System.currentTimeMillis();

//        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        BufferedReader bufferedReader = new BufferedReader(new FileReader("res/b1/small_input.txt"));
        String line;
        ArrayList<String> lineList = new ArrayList<>();
        while ((line = bufferedReader.readLine()) != null && !line.isEmpty()) {
            lineList.add(line);
        }

        System.out.format("Read %d sentences in %.2fs\n", lineList.size(), (System.currentTimeMillis() - startTime) / 1000.0f);

        System.out.println("");

        System.out.println("POS tagging sentences...");
        startTime = System.currentTimeMillis();

        MaxentTagger tagger = new MaxentTagger(MaxentTagger.DEFAULT_JAR_PATH);

        for (int i = 0; i < lineList.size(); i++) {
            lineList.set(i, tagger.tagString(lineList.get(i)));
        }

        System.out.format("POS tagged sentences in %.2fs\n", (System.currentTimeMillis() - startTime) / 1000.0f);

        System.out.println("");

        System.out.println("Pre-processing sentences...");
        startTime = System.currentTimeMillis();

        Properties properties = new Properties();
        properties.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        pipeline = new StanfordCoreNLP(properties);

        BufferedReader stopWordBufferedReader = new BufferedReader(new FileReader("res/stop_words.txt"));
        stopWordTrie = new Trie();
        while ((line = stopWordBufferedReader.readLine()) != null) {
            stopWordTrie.add(line);
        }
        stopWordBufferedReader.close();

        ArrayList<Sentence> sentenceList = new ArrayList<>();
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("res/b1/output.txt"));
        for (String s : lineList) {
            Sentence sentence = new Sentence(s);
            sentenceList.add(sentence);
            bufferedWriter.write(sentence + "\n");
        }
        bufferedWriter.write("\n");

        System.out.format("Pre-processed sentences in %.2fs\n", (System.currentTimeMillis() - startTime) / 1000.0f);

        System.out.println("");

        System.out.println("Calculating A matrix...");

        startTime = System.currentTimeMillis();
        TreeSet<String> tagSet = new TreeSet<>();
        for (Sentence sentence : sentenceList) {
            for (Word word : sentence.wordList) {
                tagSet.add(word.t);
            }
        }
        ArrayList<String> tagList = new ArrayList<>(tagSet);
        int tagCount = tagSet.size();
        HashMap<String, Integer> tagMap = new HashMap<>();
        for (int i = 0; i < tagCount; i++) {
            tagMap.put(tagList.get(i), i);
        }

        float[][] a = new float[tagCount][tagCount];
        for (Sentence sentence : sentenceList) {
            for (int i = 0; i < sentence.wordList.size() - 1; i++) {
                Word word1 = sentence.wordList.get(i),
                        word2 = sentence.wordList.get(i + 1);
                a[tagMap.get(word1.t)][tagMap.get(word2.t)]++;
            }
        }
        for (int i = 0; i < tagCount; i++) {
            int count = 0;
            for (int j = 0; j < tagCount; j++) {
                count += a[i][j];
            }

            for (int j = 0; j < tagCount; j++) {
                a[i][j] /= count;
            }
        }

        bufferedWriter.write(String.format("%-14s", "A"));
        for (String tag : tagList) {
            bufferedWriter.write(String.format("%-7s", tag));
        }
        bufferedWriter.write("\n\n");
        for (int i = 0; i < tagCount; i++) {
            bufferedWriter.write(String.format("%-14s", tagList.get(i)));
            for (int j = 0; j < tagCount; j++) {
                bufferedWriter.write(String.format("%-7.2f", a[i][j]));
            }
            bufferedWriter.write("\n\n");
        }

        System.out.format("Calculated A matrix as %dx%d in %.2fs\n", tagCount, tagCount, (System.currentTimeMillis() - startTime) / 1000.0f);

        System.out.println("");

        System.out.println("Calculating B matrix...");

        startTime = System.currentTimeMillis();
        TreeSet<String> wordSet = new TreeSet<>();
        for (Sentence sentence : sentenceList) {
            for (Word word : sentence.wordList) {
                wordSet.add(word.w);
            }
        }
        ArrayList<String> wordList = new ArrayList<>(wordSet);
        int wordCount = wordSet.size();
        HashMap<String, Integer> wordMap = new HashMap<>();
        for (int i = 0; i < wordCount; i++) {
            wordMap.put(wordList.get(i), i);
        }

        float[][] b = new float[wordCount][tagCount];
        for (Sentence sentence : sentenceList) {
            for (Word word : sentence.wordList) {
                b[wordMap.get(word.w)][tagMap.get(word.t)]++;
            }
        }
        for (int i = 0; i < wordCount; i++) {
            int count = 0;
            for (int j = 0; j < tagCount; j++) {
                count += b[i][j];
            }

            for (int j = 0; j < tagCount; j++) {
                b[i][j] /= count;
            }
        }

        bufferedWriter.write(String.format("%-14s", "B"));
        for (String tag : tagList) {
            bufferedWriter.write(String.format("%-7s", tag));
        }
        bufferedWriter.write("\n\n");
        for (int i = 0; i < wordCount; i++) {
            bufferedWriter.write(String.format("%-14s", wordList.get(i)));
            for (int j = 0; j < tagCount; j++) {
                bufferedWriter.write(String.format("%-7.2f", b[i][j]));
            }
            bufferedWriter.write("\n\n");
        }

        System.out.format("Calculated B matrix as %dx%d in %.2fs\n", wordCount, tagCount, (System.currentTimeMillis() - startTime) / 1000.0f);

        System.out.println("");

        System.out.println("Calculating Pi matrix...");

        startTime = System.currentTimeMillis();
        float[] pi = new float[tagCount];
        for (Sentence sentence : sentenceList) {
            Word word = sentence.wordList.get(0);
            pi[tagMap.get(word.t)]++;
        }
        int sentenceCount = sentenceList.size();
        for (int i = 0; i < tagCount; i++) {
            pi[i] /= sentenceCount;
        }

        bufferedWriter.write(String.format("%-14s", "Pi"));
        for (String tag : tagList) {
            bufferedWriter.write(String.format("%-7s", tag));
        }
        bufferedWriter.write("\n\n");
        bufferedWriter.write(String.format("%-14s", ""));
        for (int i = 0; i < tagCount; i++) {
            bufferedWriter.write(String.format("%-7.2f", pi[i]));
        }
        bufferedWriter.write("\n\n");

        System.out.format("Calculated Pi matrix as 1x%d in %.2fs\n", tagCount, (System.currentTimeMillis() - startTime) / 1000.0f);

        Sentence sentence = new Sentence(tagger.tagString(bufferedReader.readLine()));
        bufferedWriter.write(sentence + "\n\n");

        Word word = sentence.wordList.get(0);
        float f = pi[tagMap.get(word.t)];
        bufferedWriter.write(String.format("%.2f", f));
        for (int i = 0; i < sentence.wordList.size() - 1; i++) {
            Word word1 = sentence.wordList.get(i),
                    word2 = sentence.wordList.get(i + 1);
            float aij = a[tagMap.get(word1.t)][tagMap.get(word2.t)];
            bufferedWriter.write(String.format(" * %.2f", aij));

            f *= aij;
        }
        bufferedWriter.write(" = " + f);

        bufferedReader.close();
        bufferedWriter.close();
    }

    private static class Trie {

        private Node rootNode = new Node();

        public void add(String word) {
            Node node = rootNode;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                Node childNode = node.childNodeMap.get(c);
                if (childNode == null) {
                    childNode = new Node();
                    node.childNodeMap.put(c, childNode);
                }

                node = childNode;
            }

            node.word = true;
        }

        public boolean contains(String word) {
            Node node = rootNode;
            for (int i = 0; i < word.length(); i++) {
                Node childNode = node.childNodeMap.get(word.charAt(i));
                if (childNode == null) {
                    return false;
                }

                node = childNode;
            }

            return node.word;
        }

        private class Node {

            private boolean word;
            private TreeMap<Character, Node> childNodeMap = new TreeMap<>();

        }

    }

    private static class Word {

        private String w, t;

        public Word(String w, String t) {
            this.w = w;
            this.t = t;
        }

        @Override
        public String toString() {
            return w + "_" + t;
        }

    }

    private static class Sentence {

        private ArrayList<Word> wordList = new ArrayList<>();

        public Sentence(String s) {
            for (String w : s.split(" ")) {
                int i = w.lastIndexOf('_');
                String t = w.substring(i + 1);
                w = w.substring(0, i)
                        .toLowerCase()
                        .replaceAll("[^a-z]", "");

                if (!w.isEmpty()) {
                    pipeline.process(w)
                            .get(CoreAnnotations.SentencesAnnotation.class)
                            .get(0).get(CoreAnnotations.TokensAnnotation.class)
                            .get(0).get(CoreAnnotations.LemmaAnnotation.class);

                    if (!stopWordTrie.contains(w)) {
                        wordList.add(new Word(w, t));
                    }
                }
            }
        }

        @Override
        public String toString() {
            String s = "";
            for (Word word : wordList) {
                s += word + " ";
            }
            return s;
        }

    }

}
