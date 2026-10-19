"""What the narrator says, beat by beat.

Arabic phrases are marked [[ar:ARABIC|transliteration]] and spoken in Arabic by the same voice; Islamic
words and names in the English are spoken from Arabic script (TTS_SPELLING). Numbers are written as words
for the voice and shown as digits in the subtitles (CAPTION_NUMBERS).
"""

LINES = {
    # -- 1. what a du'a recognizer is
    "open1": "This phone was never told which du'a this is. It worked it out from the words, out of five hundred "
             "and twenty-two texts, and now it follows the recitation, line by line, word by word.",
    "shazam1": "You've probably used Shazam. Hold your phone up to a song in a cafe, and a few seconds later it tells "
               "you its name. It turns those seconds of sound into a picture of frequencies over time, and keeps only "
               "the brightest points. Pairs of points become short codes, looked up among millions of songs. The song "
               "whose codes all line up at the same moment wins.",
    "catch1": "But those points belong to one recording. Another reciter, another room, and they change completely. "
              "What every recitation shares is the words.",
    "tarteel1": "Tarteel is an app for reciting the Quran, and it listens for the words. You recite; speech recognition "
                "writes down what you said; that text is searched across the Quran to find the surah and the ayah; and "
                "from then on, each word lights up as you say it. Hide the page, and the words appear only as you "
                "recite them, and it can mark a word you missed.",
    "tarteel2": "Its founders started with an earlier app that used a phone's own speech recognition. That was good "
                "enough to find an ayah, but it made more mistakes than the reciters it was meant to check. So they "
                "crowdsourced recordings, curated more than seventy-five thousand minutes of recitation, and trained "
                "models of their own. They published small ones too, and one of those is where our phone model began.",
    "took1": "From Shazam: search everything at once, from anywhere in a recording, and trust only matches that line "
             "up in time. From Tarteel: listen for the words, and follow them. Du'as add their own problems: many "
             "texts share the same phrases, and people pause, go back and talk between lines.",
    # -- 2. data
    "data1": "The data comes in three kinds. The texts: five hundred and twenty-two du'as, ziyarat and munajat, about "
             "a hundred and forty thousand words. Recitations: DuaPlayer's are timed line by line by hand, so they "
             "are the ruler everything is measured against.",
    "data2": "For more voices, we searched YouTube, Shiavoice, Aparat, SoundCloud and the Internet Archive for every "
             "text: five thousand eight hundred hours. A large letter model "
             "listened to every file, found which text and which line it was, and kept only the lines that fit: "
             "about fourteen hundred hours, in about thirty-five hundred voices.",
    "data3": "And ordinary voices, because reciters aren't the people holding the phone: crowd-sourced recitations, "
             "and synthetic ones, a voice-cloning model reading du'as in three hundred and twelve volunteers' "
             "voices. Test voices were removed by name and by sound, so the results are on voices the models never "
             "heard.",
    # -- 3. models
    "models1": "Two models listen at once, each with one job. Both read the same picture of sound.",
    "whisper1": "The first is Whisper, OpenAI's speech recogniser. One half listens, the other writes. It reads the "
                "last six seconds and writes down the words. On the phone it's Whisper base, seventy-four million "
                "parameters, grown from Tarteel's Quran model; on the server, Large v3 Turbo, eight hundred and nine "
                "million.",
    "ctc1": "The second is a letter model. Fifty times a second, it answers one question: which letter is being said "
            "now, or none. A drawn-out word is one letter held over many frames; a pause is all blanks. It's Whisper "
            "base's listening half, with one layer onto the alphabet: twenty million parameters.",
    "train1": "Whisper was fine-tuned on du'a windows labelled with the du'a's own text, never a machine's guess at "
              "it. Every window was altered on the way in: faster, slower, a deeper or higher voice, noise, an "
              "echoing room, a hall through loudspeakers with people talking, stripes of the picture blanked out.",
    "distil1": "Then each small model learnt from a big one: the phone's Whisper from the server's, hearing a noisy "
               "copy of what the big one hears clean; the letter model from a three-hundred-million-parameter "
               "wav2vec2.",
    # -- 4. recognition
    "rec1": "Now, recognition. Every second or two, Whisper writes down the last six seconds. Here, it hears this.",
    "rec1b": "The vowel marks and the spaces are dropped, leaving a string of letters.",
    "rec2": "That string is slid along every word of all five hundred and twenty-two texts: how many letters would "
            "have to change for it to end right here? A trick by Gene Myers works on sixty-four letters at once, "
            "so the whole library takes about three milliseconds.",
    "rec3": "One window isn't enough. [[ar:اللَّهُمَّ|Allāhumma]] is in four hundred and twelve of the texts, "
            "[[ar:اللَّهُمَّ إِنِّي أَسْأَلُكَ|Allāhumma innī asʾaluka]] in a hundred and six. And since the "
            "matching forgives a few wrong letters, for a second Du'a Jawshan Kabir leads: its hundred sections "
            "open with almost these words.",
    "rec4": "So a tracker keeps a probability on every word of every du'a, the way a map keeps your GPS dot on roads "
            "you could have reached. Each update moves them on at the reader's pace, then weighs them by the "
            "matches. With the next window only Kumayl fits, and once a du'a holds seventy percent, its name is "
            "shown.",
    "rec5": "That's Shazam's idea, matches that line up in time, applied to words as Tarteel does: it needs no copy "
            "of the recording, only the text.",
    "results1": "On a test bench of voices held out of training, read every way people read, the phone is on the right "
                "line ninety-one percent of the time and on the right word seventy-six, and names the du'a within ten "
                "seconds of its first words no other text shares ninety-four percent of the time.",
    "results2": "It's weakest far from the phone, in a big hall, and with du'as it doesn't have. For halls, we built a "
                "masjid in software: a PA, loudspeakers in measured rooms, people talking. The phone went from "
                "fifty-five to sixty-seven percent on the line there, most of it from training in simulated halls; "
                "the server reaches eighty-one.",
    # -- 5. following
    "follow1": "Whisper's text comes late and has no timing inside it, so the highlight comes from the letter model. "
               "A decoder keeps a probability on every letter of the du'a: the tracker's idea again, at a finer "
               "scale.",
    "follow2": "Reading on is free. Starting the line again costs a little, going back a few lines more, skipping "
               "ahead more still, so it takes clear letters to move far. Here, the reader goes back two lines.",
    "follow3": "A pause makes no letters, and while no voice is heard, stray letters cost more, so the highlight "
               "stays on the last word said. Talk and a salawat between lines have places of their own to go, and "
               "it enters the next line only once it hears that line's words.",
    "follow4": "When someone goes back or skips, it follows within three seconds eighty-four percent of the time; "
               "through pauses, talk and salawat, it holds still ninety-one percent.",
    "cost1": "All of it runs in the phone's browser, so no audio leaves the phone. A letter-model step takes about "
             "seven milliseconds in Chrome on a desktop, because it reads only the new audio; Whisper takes under two "
             "seconds a window on a phone, so it runs every two.",
    # -- 6. conclusion
    "recap1": "So, from the top. Every two seconds, Whisper writes down the last six; the words are matched against "
              "all five hundred and twenty-two texts in milliseconds; the tracker names the du'a once the matches "
              "agree over time. Fifty times a second, the letter model hears letters, and the decoder places them: "
              "the line, and the word.",
    "credits1": "It stands on other people's work: the reciters, the sites that publish their recordings and texts, "
                "and the open models, data and tools on screen.",
}

# The model and data lines bring the most new names per second: said a little slower.
SLOW: set[str] = set()

# Islamic words and names, written for the voice in Arabic script so it says them as an Arabic speaker would.
TTS_SPELLING = {
    "Du'a Jawshan Kabir": "دعاء الجوشن الكبير",
    "Kumayl": "كميل",
    "Tarteel's": "ترتيل",
    "Tarteel": "ترتيل",
    "Quran": "القرآن",
    "surah": "سورة",
    "ayah": "آية",
    "ziyarat": "زيارات",
    "munajat": "مناجاة",
    "salawat": "الصلوات",
    "masjid": "مسجد",
    "du'as": "أدعية",
    "du'a's": "الدعاء",
    "du'a": "دعاء",
}

# Subtitles show numbers as digits; the voice says the words.
CAPTION_NUMBERS = [
    ("five hundred and twenty-two", "522"), ("a hundred and forty thousand", "140,000"),
    ("five thousand eight hundred hours", "5,800 hours"), ("fourteen hundred hours", "1,400 hours"),
    ("thirty-five hundred voices", "3,500 voices"), ("three hundred and twelve", "312"),
    ("seventy-five thousand minutes", "75,000 minutes"), ("seventy-four million", "74 million"), ("eight hundred and nine million", "809 million"),
    ("twenty million", "20 million"), ("three-hundred-million-parameter", "300-million-parameter"),
    ("sixty-four", "64"), ("four hundred and twelve", "412"), ("a hundred and six", "106"),
    ("seventy percent", "70%"), ("ninety-one percent", "91%"), ("seventy-six,", "76%,"),
    ("ninety-four percent", "94%"), ("fifty-five to sixty-seven percent", "55% to 67%"),
    ("reaches eighty-one", "reaches 81%"), ("eighty-four percent", "84%"), ("ninety-one percent", "91%"),
    ("ten seconds", "10 seconds"), ("three seconds", "3 seconds"), ("six seconds", "6 seconds"),
    ("seven milliseconds", "7 milliseconds"), ("three milliseconds", "3 milliseconds"),
    ("Large v3 Turbo", "large-v3-turbo"), ("Whisper base", "Whisper-base"),
]
