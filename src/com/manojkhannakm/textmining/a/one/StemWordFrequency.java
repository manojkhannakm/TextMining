package com.manojkhannakm.textmining.a.one;

import edu.northwestern.at.utils.corpuslinguistics.stemmer.LancasterStemmer;

import java.io.*;
import java.util.Map;
import java.util.TreeMap;

/**
 * @author Manoj Khanna
 */

public class StemWordFrequency {

    public static void main(String[] args) throws IOException {
        System.out.println("Reading input.txt...");
        long startTime = System.currentTimeMillis();

//        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        BufferedReader bufferedReader = new BufferedReader(new FileReader("res/a1/small_input.txt"));
        String line;
        StringBuilder stringBuilder = new StringBuilder();
        while ((line = bufferedReader.readLine()) != null && !line.isEmpty()) {
            stringBuilder.append(line);
        }
        bufferedReader.close();

        String[] words = stringBuilder.toString().toLowerCase().split("[^A-Za-z]+");
        System.out.format("Read %d words in %.2fs\n", words.length, (System.currentTimeMillis() - startTime) / 1000.0f);

        System.out.println("");

        bufferedReader = new BufferedReader(new FileReader("res/stop_words.txt"));
        Trie stopWordTrie = new Trie();
        while ((line = bufferedReader.readLine()) != null) {
            stopWordTrie.add(line);
        }
        bufferedReader.close();
        stopWordTrie.print();

        System.out.println("");

        System.out.println("Removing stop words...");
        startTime = System.currentTimeMillis();

        int stopWordCount = 0;
        for (int i = 0; i < words.length; i++) {
            if (stopWordTrie.contains(words[i])) {
                words[i] = null;
                stopWordCount++;
            }
        }

        System.out.format("Removed %d stop words in %.2fs\n", stopWordCount, (System.currentTimeMillis() - startTime) / 1000.0f);

        System.out.println("");

        System.out.println("Writing output...");

        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("res/a1/output.txt"));
        for (String word : words) {
            if (word != null) {
                bufferedWriter.write(word + " ");
            }
        }
        bufferedWriter.close();

        System.out.println("");

        System.out.println("Finding stem words...");

        LancasterStemmer stemmer = new LancasterStemmer();
        TreeMap<String, Integer> stemWordMap = new TreeMap<>();
        for (String word : words) {
            if (word != null) {
                String stemWord = stemmer.stem(word);
                Integer stemWordCount = stemWordMap.get(stemWord);
                stemWordMap.put(stemWord, stemWordCount == null ? 1 : stemWordCount + 1);
            }
        }

        for (Map.Entry<String, Integer> entry : stemWordMap.entrySet()) {
            System.out.println(entry.getKey() + " -> " + entry.getValue());
        }

        System.out.println("Found " + stemWordMap.size() + " stem words");
    }

    private static class Trie {

        private Node rootNode = new Node('\0');

        public void add(String word) {
            Node node = rootNode;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                Node childNode = node.childNodeMap.get(c);
                if (childNode == null) {
                    childNode = new Node(c);
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

        public void print() {
            System.out.println("\\0");

            for (Node node : rootNode.childNodeMap.values()) {
                node.print(1, new char[999]);
            }
        }

        private class Node {

            private char c;
            private boolean word;
            private TreeMap<Character, Node> childNodeMap = new TreeMap<>();

            public Node(char c) {
                this.c = c;
            }

            private void print(int level, char[] chars) {
                chars[level - 1] = c;

                System.out.format("%" + level + "s", "");
                System.out.print(c);
                if (word) {
                    System.out.print(" -> " + new String(chars, 0, level));
                }
                System.out.println("");

                for (Node node : childNodeMap.values()) {
                    node.print(level + 1, chars);
                }
            }

        }

    }

}
