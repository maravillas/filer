(ns filer.core
  (:use [clojure.java.io])
  (:require [clojure.string :as s])
  (:import [java.io StringWriter FilenameFilter]
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
  (let [classes (map #(make-class % root) (populated-dirs root))]
    (map train-class classes)))

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

(defn classify
  [all-freqs classes path]
  (let [text (file-freqs path)
        prob-fn #(prob all-freqs (count classes) %1 %2)]
    (map (fn [c]
           (vector (:name c) (calculate-class #(prob-fn c %) text c)))
         classes)))

(defn select-class
  [scores]
  (first
   (reduce (fn [[c1 v1] [c2 v2]] (if (> v1 v2) [c1 v1] [c2 v2]))
           scores)))

(defn test-doc
  [all-freqs classes class doc]
  (let [scores (classify all-freqs classes doc)]
    {:correct (.endsWith (.getAbsolutePath (.getParentFile doc)) (select-class scores))
     :doc doc
     :scores scores}))

(defn test-class
  [all-freqs classes class]
  (map #(test-doc all-freqs classes class %) (:test-data class)))

(defn summarize-class-results
  [class results]
  (let [correct (count (filter :correct results))
        total (count results)]
    {:class (:name class)
     :correct correct
     :incorrect (- total correct)
     :accuracy (if (zero? total) 0 (* 100.0 (/ correct total)))}))

(defn test-all
  [all-freqs classes]
  (map #(summarize-class-results % (test-class all-freqs classes %)) classes))

