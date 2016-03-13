import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.util.Properties;

/**
 * @author Manoj Khanna
 */

public class Test {

    public static void main(String[] args) {
        Properties properties = new Properties();
        properties.setProperty("annotators", "tokenize, ssplit, pos, lemma");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(properties);
        Annotation document = pipeline.process("run ran runs running");
        for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                System.out.println(token.lemma());
            }
        }

//        System.out.println(Normalizer.normalize("", Normalizer.Form.NFKC));

//        System.out.println(new MaxentTagger(MaxentTagger.DEFAULT_JAR_PATH).tagString("My email is manojkhannakm@gmail.com."));
    }

}
