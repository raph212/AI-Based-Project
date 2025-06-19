import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Use the microphone as source
with sr.Microphone() as source:
    print("🎤 Please speak something...")
    recognizer.adjust_for_ambient_noise(source)
    audio_data = recognizer.listen(source)

    print("🔍 Recognizing...")

    try:
        # Convert speech to text using Google Web Speech API
        text = recognizer.recognize_google(audio_data)
        print("📝 You said:", text)
    except sr.UnknownValueError:
        print("❌ Could not understand the audio")
    except sr.RequestError as e:
        print("⚠️ Could not request results; {0}".format(e))
