##UTILITY FUNCTION

def convert_mp3_to_wav(audio:str) -> str:  
    """Convert an input MP3 audio track into a WAV file.

    Args:
        audio (str): An input audio track.

    Returns:
        [str]: WAV filename.
    """
    if audio[-3:] == "mp3":
        wav_audio = audio[:-3] + "wav"
        if not Path(wav_audio).exists():
                subprocess.check_output(f"ffmpeg -i {audio} {wav_audio}", shell=True)
        return wav_audio
    
    return audio

def plot_spectrogram_and_picks(track:np.ndarray, sr:int, peaks:np.ndarray, onset_env:np.ndarray) -> None:
    """[summary]

    Args:
        track (np.ndarray): A track.
        sr (int): Aampling rate.
        peaks (np.ndarray): Indices of peaks in the track.
        onset_env (np.ndarray): Vector containing the onset strength envelope.
    """
    times = librosa.frames_to_time(np.arange(len(onset_env)),
                            sr=sr, hop_length=HOP_SIZE)

    plt.figure()
    ax = plt.subplot(2, 1, 2)
    D = librosa.stft(track)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                            y_axis='log', x_axis='time')
    plt.subplot(2, 1, 1, sharex=ax)
    plt.plot(times, onset_env, alpha=0.8, label='Onset strength')
    plt.vlines(times[peaks], 0,
            onset_env.max(), color='r', alpha=0.8,
            label='Selected peaks')
    plt.legend(frameon=True, framealpha=0.8)
    plt.axis('tight')
    plt.tight_layout()
    plt.show()

def load_audio_picks(audio, duration, hop_size):
    """[summary]

    Args:
        audio (string, int, pathlib.Path or file-like object): [description]
        duration (int): [description]
        hop_size (int): 

    Returns:
        tuple: Returns the audio time series (track) and sampling rate (sr), a vector containing the onset strength envelope
        (onset_env), and the indices of peaks in track (peaks).
    """
    try:
        track, sr = librosa.load(audio, duration=duration)
        onset_env = librosa.onset.onset_strength(track, sr=sr, hop_length=hop_size)
        peaks = librosa.util.peak_pick(onset_env, 10, 10, 10, 10, 0.5, 0.5)
    except Error as e:
        print('An error occurred processing ', str(audio))
        print(e)

    return track, sr, onset_env, peaks

------------------------------------------------------------------------------------------

##FINGERPRINT HASHING

#Function to index every song in our dataset
def track_vocabulary(tracks):
    voc={}
    i=1
    for track in tracks:
        name = os.path.basename(track)
        name=name[3:-4]

        path = os.path.normpath(track)
        final=path[14:]
        lst=final.partition('\\')
        author=lst[0]
        voc[i]=(name, author)
        i=i+1      
    return voc

#Function to generate pseudorandom numbers later used in our hash
def random_number(n_permutation):
    random.seed(10010000) #random seed to generate PSEUDO-random numbers
    #our module is one of the Mersenne prime number, in particular the 8th 
    c=2147483647
    #as the ceil of the random number generator we had choosen the dimension of the unsigned long
    long=4294967295 
    a=random.sample(range(0, long), n_permutation)
    b=random.sample(range(0, long), n_permutation)
    return a, b, c

------------------------------------------------------------------------------------------

##PEAKS SET

#Function to create a list of all unique peaks found in our dataset
def create_peaks_set(tracks):
    all_peaks=[]
    for track in tracks:
        _, _, _, peaks = load_audio_picks(track, DURATION, HOP_SIZE)
        for p in peaks:
            if p not in all_peaks:
                all_peaks.append(p)
                #print(p)
    all_peaks=sorted(all_peaks)
    return all_peaks

------------------------------------------------------------------------------------------

##Shingles table and signature

#Function to create a matrix of 0s and 1s as explained in the notebook
def shingles_table(all_peaks, tracks, lens):

    matrix=np.zeros((len(all_peaks), lens), dtype=int)
    
    for i, track in enumerate(tracks):
        _, _, _, peaks = load_audio_picks(track, DURATION, HOP_SIZE)

        for j, p in enumerate(all_peaks):
            if p in peaks:
                matrix[j][i]=1
    
    return matrix

#Minhashing our signatures
def signature_min_hash(matrix, n_permutation):

    signat=[]
    
    for _ in range(n_permutation):
        np.random.seed(10010000)
        np.random.shuffle(matrix)
        #compute signature
        row=[]
        #extract the first element of the audio
        for i in range(0, len(matrix[0])):
            #print(i)
            element=0
            element=np.where(matrix[:, i]==1)[0][0]
            row.append(element)
        signat.append(row)

    return signat

------------------------------------------------------------------------------------------

##Creating buckets

#Our custom hash function
def custom_hash(A, n_band):
    a, b, c=random_number(n_band) #utilizing our pseudorandom values
    a=np.asarray(a)
    a=np.asarray(b)
    A=np.asarray(A)
    return (np.sum(a*A+b))%c


#Creating our two dictionaries with indexes and buckets
def creating_bucket(matrix):

    bucket={} #dict with [ keys = song_index] : [values = list of bucket]
    inverted_bucket={}  #dict with [keys = list of bucket] : [ values = song_index] 
    n_band=8

    for i in range(len(matrix[0])):
        for j in range(0, N_PERMUTATION_LSH, n_band):
            key=[]
            for k in range(j, j+8):
                key.append(matrix[k][i])
            
            hash=custom_hash(key, n_band)


            if i not in bucket:
                bucket[i]=[]

            bucket[i].append(hash)

            #-----------------#

            if hash not in inverted_bucket:
                inverted_bucket[hash]=[]

            inverted_bucket[hash].append(i)


    return  bucket, inverted_bucket

------------------------------------------------------------------------------------------

##Query preprocessing

#Simple function to compute Jaccard Similarity
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union


#Function to actually compare our query to the possible matches
def similarity(song_buckets_query, song_buckets, bucket, threshold):
    jaccard=[]
    for i in song_buckets_query:
        list_index_matched=[]
        values=song_buckets_query[i]
        for v in values:
            try:
                song_index=bucket[v]
                if song_index  not in list_index_matched:

                    list_index_matched.append(song_index)
                    set_buckets=song_buckets[song_index[0]]
                    j=jaccard_similarity(set_buckets, values)

                    if j > threshold: #We only return results if the similarity is larger than the threshold
                        jaccard.append((i, j, song_index))
            except:
                pass
            

    return jaccard