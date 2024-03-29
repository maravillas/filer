(ns filer.core
  (:use [clojure.java.io])
  (:require [clojure.string :as s])
  (:import [java.io StringWriter FilenameFilter]
           [java.math RoundingMode MathContext]
           [org.apache.pdfbox.util PDFTextStripper]
           [org.apache.pdfbox.pdmodel PDDocument]))

(defn pdf?
  [name]
  (.endsWith name ".pdf"))

(def pdf-filter
  (proxy [FilenameFilter] []
    (accept [dir name] (pdf? name))))

(defn extract-text
  [pdf]
  (let [text-writer (StringWriter.)]
    (with-open [pdf (PDDocument/load pdf)]
      (.writeText (PDFTextStripper.) pdf text-writer))
    (str text-writer)))

(defn tokenize
  [text]
  (s/split text #"\s+"))

(defn file-freqs
  [path]
  (-> path
      file
      extract-text
      .toLowerCase
      tokenize
      frequencies))

(defn sum-freqs
  [freqs]
  (reduce #(merge-with + %1 %2) freqs))

(defn class-freqs
  [class]
  (when (seq (:training-data class))
    (sum-freqs (map file-freqs (:training-data class)))))

(defn train-class
  [class]
  (assoc class :freqs (class-freqs class)))

(defn has-pdfs?
  [dir]
  (> (count (.listFiles dir pdf-filter))
     0))

(defn populated-dirs
  [root]
  (letfn [(helper [dir]
            (let [directories (filter #(.isDirectory %) (.listFiles dir))
                  classes (filter has-pdfs? directories)
                  classes (map #(s/replace (.getAbsolutePath %) (.getAbsolutePath root) "") classes)]
              (concat classes (mapcat helper directories))))]
    (helper root)))

(defn partition-data
  [class]
  (let [files (filter #(not (.isDirectory %)) (.listFiles (:path class) pdf-filter))
        [training test] (split-at (/ (count files) 2) (shuffle files))]
    {:training-data training
     :test-data test}))

(defn make-class
  [name root]
  (let [class {:name name
               :path (file root name)}]
    (merge class (partition-data class))))

(defn train
  [root]
  (let [classes (map #(make-class % root) (populated-dirs root))
        trained-classes (map train-class classes)
        all-freqs (sum-freqs (map :freqs trained-classes))]
    {:all-freqs all-freqs
     :classes trained-classes}))

(defn prob
  "Probability that word occurs in a class."
  [all-freqs class-count class word]
  (/ (inc (or ((:freqs class) word) 0))
     (+ (or (all-freqs word) 0) class-count)))

(defn calculate-class
  [prob-fn text class]
  (reduce +
          (map (fn [[word count]] (* count (Math/log (prob-fn word))))
               text)))

(defn split-pow
  "Calculate a non-integer BigDecimal power by calculating the exponent's
   remainder separately.

   b^(i+r) == b^i*b^r"
  [base exp]
  (let [base (BigDecimal. base)
        exp (BigDecimal. exp)
        r (.remainder exp BigDecimal/ONE)
        i (.intValueExact (.subtract exp r))
        int-pow (.pow base (int i) MathContext/DECIMAL64)
        rem-pow (BigDecimal. (Math/pow base r))]
    (.multiply int-pow rem-pow)))

(defn truncate-big-decimal
  [d n]
  (-> d
      (.setScale n RoundingMode/DOWN)
      .doubleValue))

(defn relative-scores
  [scores]
  (let [sum (reduce + (vals scores))]
    (zipmap (keys scores)
            (map #(truncate-big-decimal (.divide % sum MathContext/DECIMAL64) 4)
                 (vals scores)))))

(defn classify
  [{:keys [all-freqs classes] :as training-data} path]
  (let [text (file-freqs path)
        prob-fn #(prob all-freqs (count classes) %1 %2)
        scores (into {} (map (fn [c]
                               [(:name c) (split-pow Math/E (calculate-class #(prob-fn c %) text c))])
                             classes))]
    (relative-scores scores)))

(defn classify-dir
  [db dir]
  (let [files (.listFiles dir pdf-filter)]
    (map (fn [f] {:filename (.getName f)
                  :path (.getPath f)
                  :scores (classify db f)})
         files)))

(defn select-class
  [scores]
  (first
   (reduce (fn [[c1 v1] [c2 v2]] (if (> v1 v2) [c1 v1] [c2 v2]))
           scores)))

(defn test-doc
  [training-data class doc]
  (let [scores (classify training-data doc)]
    {:correct (.endsWith (.getAbsolutePath (.getParentFile doc)) (select-class scores))
     :doc doc
     :scores scores}))

(defn test-class
  [training-data class]
  (map #(test-doc training-data class %) (:test-data class)))

(defn summarize-class-results
  [class results]
  (let [correct (count (filter :correct results))
        total (count results)]
    {:class (:name class)
     :correct correct
     :incorrect (- total correct)
     :accuracy (if (zero? total) 0 (* 100.0 (/ correct total)))}))

(defn test-all
  [{:keys [all-freqs classes] :as training-data}]
  (map #(summarize-class-results % (test-class training-data %)) classes))


