import streamlit as st
import numpy as np
import json
import pickle

def load_vocab(filename):
    with open(filename) as f:
        dict = json.loads(f.read())
    return dict

def load_features(filename):
    array = np.loadtxt(filename, dtype = str)
    return array

def return_feature_matrix(dataset,features,vocab):
    feature_matrix=np.zeros((len(dataset),len(features)),int) #CREATING A LIST OF ZERO
    for i in range(len(dataset)):
        for j in dataset[i].split():
            if j in features:
                feature_matrix[i][vocab[j]]+=1
    return feature_matrix


def main():
    html_temp = """
        <div style="background-color:purple ;padding:10px">
        <h2 style="color:white;text-align:left;">Text Classification</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ["Atheist Resources", "Navy Scientific Visualisation and Virtual Reality Seminar", "Saturn's Pricing Policy", "Gov't break-ins"]

    option = st.sidebar.selectbox('Choose a document', activities)
    with st.sidebar.expander("List of 20 Newsgroups"):
        st.write(
            """
            - comp.graphics
            - comp.os.ms-windows.misc
            - comp.sys.ibm.pc.hardware
            - comp.sys.mac.hardware
            - comp.windows.x
            - rec.autos
            - rec.motorcycles
            - rec.sport.baseball
            - rec.sport.hockey
            - sci.crypt
            - sci.electronics
            - sci.med
            - sci.space
            - misc.forsale
            - talk.politics.misc
            - talk.politics.guns
            - talk.politics.mideast
            - talk.religion.misc
            - alt.atheism
            - soc.religion.christian
            """
        )
    # 1, 3, rec.autos, sci.space
    st.write(f"Document Selected :  {option}")
    file = option
    data = []
    with open("docs" + '/' + file, 'r', errors='ignore') as fileobj:
        data.append(fileobj.read())
    data=np.array(data)
    print(data)
    vocab = load_vocab("vocabulary.json")
    features = load_features("features.txt")
    input = return_feature_matrix(data,features,vocab)
    loaded_model = pickle.load(open("model.pkl", 'rb'))
    if st.button('Classify'):
        st.success(loaded_model.predict(input)[0])

if __name__ == '__main__':
    main()
