package com.manojkhannakm.textmining.a.two;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import java.io.*;
import java.util.ArrayList;
import java.util.Properties;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Latent Semantic Analysis
 *
 * @author Manoj Khanna
 */

public class LSA {

    private static StanfordCoreNLP pipeline;
    private static Trie stopWordTrie;

    public static void main(String[] args) throws IOException {
        System.out.println("Reading input.txt...");
        long startTime = System.currentTimeMillis();

//        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        BufferedReader bufferedReader = new BufferedReader(new FileReader("res/a2/input.txt"));
        String line;
        ArrayList<String> lineList = new ArrayList<>();
        while ((line = bufferedReader.readLine()) != null && !line.isEmpty()) {
            lineList.add(line);
        }

        System.out.format("Read %d sentences in %.2fs\n", lineList.size(), (System.currentTimeMillis() - startTime) / 1000.0f);

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
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("res/a2/output.txt"));
        for (String s : lineList) {
            Sentence sentence = new Sentence(s);
            sentenceList.add(sentence);
            bufferedWriter.write(sentence + "\n");
        }
        bufferedWriter.write("\n");

        System.out.format("Pre-processed sentences in %.2fs\n", (System.currentTimeMillis() - startTime) / 1000.0f);

        System.out.println("");

        System.out.println("Constructing term-document matrix...");
        startTime = System.currentTimeMillis();

        TreeSet<String> wordSet = new TreeSet<>();
        for (Sentence sentence : sentenceList) {
            for (String word : sentence.wordList) {
                wordSet.add(word);
            }
        }
        ArrayList<String> wordList = new ArrayList<>(wordSet);

        int wordCount = wordList.size(),
                sentenceCount = sentenceList.size();
        double[][] a = new double[wordCount][sentenceCount];
        for (int i = 0; i < sentenceList.size(); i++) {
            for (String word : sentenceList.get(i).wordList) {
                a[wordList.indexOf(word)][i]++;
            }
        }

        bufferedWriter.write(String.format("%-14s", "A"));
        for (int i = 1; i <= sentenceCount; i++) {
            bufferedWriter.write(String.format("S%-6d", i));
        }
        bufferedWriter.write("\n\n");
        for (int i = 0; i < wordCount; i++) {
            bufferedWriter.write(String.format("%-14s", wordList.get(i)));
            for (int j = 0; j < sentenceCount; j++) {
                bufferedWriter.write(String.format("%-7d", (int) a[i][j]));
            }
            bufferedWriter.write("\n\n");
        }

        System.out.format("Constructed term-document matrix as %dx%d in %.2fs\n", wordCount, sentenceCount, (System.currentTimeMillis() - startTime) / 1000.0f);

        System.out.println("");

        System.out.println("Decomposing term-document matrix...");

        RealMatrix aMatrix = MatrixUtils.createRealMatrix(a);
        SingularValueDecomposition svd = new SingularValueDecomposition(aMatrix);
        int k = Integer.parseInt(bufferedReader.readLine());
        RealMatrix uMatrix = svd.getU().getSubMatrix(0, wordCount - 1, 0, k - 1),
                sMatrix = svd.getS().getSubMatrix(0, k - 1, 0, k - 1),
                vMatrix = svd.getVT().getSubMatrix(0, k - 1, 0, sentenceCount - 1),
                usMatrix = uMatrix.multiply(sMatrix),
                svMatrix = sMatrix.multiply(vMatrix);

        bufferedWriter.write(String.format("%-14s", "U"));
        for (int i = 1; i <= k; i++) {
            bufferedWriter.write(String.format("%-7d", i));
        }
        bufferedWriter.write("\n\n");
        for (int i = 0; i < wordCount; i++) {
            bufferedWriter.write(String.format("%-14s", wordList.get(i)));
            for (int j = 0; j < k; j++) {
                bufferedWriter.write(String.format("%-7.2f", uMatrix.getData()[i][j]));
            }
            bufferedWriter.write("\n\n");
        }

        bufferedWriter.write(String.format("%-14s", "S"));
        for (int i = 1; i <= k; i++) {
            bufferedWriter.write(String.format("%-7d", i));
        }
        bufferedWriter.write("\n\n");
        for (int i = 0; i < k; i++) {
            bufferedWriter.write(String.format("%-14d", i + 1));
            for (int j = 0; j < k; j++) {
                bufferedWriter.write(String.format("%-7.2f", sMatrix.getData()[i][j]));
            }
            bufferedWriter.write("\n\n");
        }

        bufferedWriter.write(String.format("%-14s", "V"));
        for (int i = 1; i <= sentenceCount; i++) {
            bufferedWriter.write(String.format("S%-6d", i));
        }
        bufferedWriter.write("\n\n");
        for (int i = 0; i < k; i++) {
            bufferedWriter.write(String.format("%-14d", i + 1));
            for (int j = 0; j < sentenceCount; j++) {
                bufferedWriter.write(String.format("%-7.2f", vMatrix.getData()[i][j]));
            }
            bufferedWriter.write("\n\n");
        }

        bufferedWriter.write(String.format("%-14s", "US"));
        for (int i = 1; i <= k; i++) {
            bufferedWriter.write(String.format("%-7d", i));
        }
        bufferedWriter.write("\n\n");
        for (int i = 0; i < wordCount; i++) {
            bufferedWriter.write(String.format("%-14s", wordList.get(i)));
            for (int j = 0; j < k; j++) {
                bufferedWriter.write(String.format("%-7.2f", usMatrix.getData()[i][j]));
            }
            bufferedWriter.write("\n\n");
        }

        bufferedWriter.write(String.format("%-14s", "SV"));
        for (int i = 1; i <= sentenceCount; i++) {
            bufferedWriter.write(String.format("S%-6d", i));
        }
        bufferedWriter.write("\n\n");
        for (int i = 0; i < k; i++) {
            bufferedWriter.write(String.format("%-14d", i + 1));
            for (int j = 0; j < sentenceCount; j++) {
                bufferedWriter.write(String.format("%-7.2f", svMatrix.getData()[i][j]));
            }
            bufferedWriter.write("\n\n");
        }

        System.out.format("Decomposed term-document matrix in %.2fs\n", (System.currentTimeMillis() - startTime) / 1000.0f);

        bufferedReader.readLine();

        Sentence sentence = new Sentence(bufferedReader.readLine());
        bufferedWriter.write(sentence + "\n\n");

        RealMatrix qMatrix = MatrixUtils.createRealMatrix(1, k);
        for (String word : sentence.wordList) {
            int i = wordList.indexOf(word);
            if (i >= 0) {
                qMatrix = qMatrix.add(usMatrix.getRowMatrix(i));
            }
        }
        qMatrix = qMatrix.scalarMultiply(1.0 / sentence.wordList.size());

        bufferedWriter.write(String.format("%-14s", "Q"));
        for (int i = 1; i <= k; i++) {
            bufferedWriter.write(String.format("%-7d", i));
        }
        bufferedWriter.write("\n\n");
        bufferedWriter.write(String.format("%-14d", 1));
        for (int j = 0; j < k; j++) {
            bufferedWriter.write(String.format("%-7.2f", qMatrix.getData()[0][j]));
        }
        bufferedWriter.write("\n\n");

        for (int i = 0; i < sentenceCount; i++) {
            double[][] sv = svMatrix.getData(),
                    q = qMatrix.getData();

            double x = 0.0;
            for (int j = 0; j < k; j++) {
                x += sv[j][i] * q[0][j];
            }

            double y1 = 0.0;
            for (int j = 0; j < k; j++) {
                y1 += sv[j][i] * sv[j][i];
            }
            y1 = Math.sqrt(y1);

            double y2 = 0.0;
            for (int j = 0; j < k; j++) {
                y2 += q[0][j] * q[0][j];
            }
            y2 = Math.sqrt(y2);

            bufferedWriter.write(String.format("cos(S%d, Q) = %.4f\n", i + 1, x / (y1 * y2)));
        }

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

    private static class Sentence {

        private ArrayList<String> wordList = new ArrayList<>();

        public Sentence(String s) {
            for (CoreMap sentence : pipeline.process(s).get(CoreAnnotations.SentencesAnnotation.class)) {
                for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                    String word = token.get(CoreAnnotations.LemmaAnnotation.class)
                            .toLowerCase()
                            .replaceAll("[^a-z]", "");
                    if (!word.isEmpty()
                            && !stopWordTrie.contains(word)) {
                        wordList.add(word);
                    }
                }
            }
        }

        @Override
        public String toString() {
            String s = "";
            for (String word : wordList) {
                s += word + " ";
            }
            return s;
        }

    }

}
