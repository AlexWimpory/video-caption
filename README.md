# video-caption
Audio pipeline performs the following steps to subtitle a video
* Extract audio – Create a wav file from the video
* Speech recogniser – Use a model to recognise speech and create a list of words with timings
* NLP  – Perform NER and dependency parsing on the output of the speech recogniser
* Train model  – A sound classification model is trained beforehand, which can be repeatedly used in the audio pipeline
* Split file  – Divide the audio into overlapping time periods for the sound classifier to analyse
* Sound predictor – Run the sound classification model on each period and find the best matching results to remove the overlap and create a single timeline of sounds
* Subtitle file/Validation  – Combine the speech recognition results, NLP results and the sound predictor results into a single subtitle file.  Spurious results are then filtered out
* Add to video – Subtitles are burnt back into the original video or they can be added in the media player used

Audio trainers can be used to create a TensorFlow model for sound classification or a SpaCy model for NER

Audio utils contains various modules for different applications such as generating noise