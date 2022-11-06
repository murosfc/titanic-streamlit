import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#Função que carrega e treina o modelo
def randomForest():
    #Obtenão dos dados
    df = pd.read_csv('titanic.csv') #Aquivo já tratado e exportado do titanicBI.py
    y = df['Survived']
    X = df.drop(['Survived'], axis=1)   
    #Divisão entre treino e teste
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=101)
    #RF classifier
    clf = RandomForestClassifier(n_estimators  = 30)
    clf = clf.fit(X, y)    
    reg = RandomForestRegressor(n_estimators = 1000, min_samples_leaf = 1, random_state = 101)
    return float(clf.score(X,y)), reg.fit(X_train, y_train)  

#Função que calcula a previsão conforme dados do usuário
def getResult(modelo, passageiro):
    return float(modelo.predict(passageiro))

#APP
#obter score e modelo
score, modelo = randomForest()

##main
st.subheader('Simule sua chance sobrevivência ao naufrágio do Titanic')
st.text('\n')
st.markdown('**Informações importantes:**')
st.caption('Considere os valores dos tickets:')
st.caption('1º classe: de USD 4.000 (beliche) à USD 115.000 (suíte)')
st.caption('2º classe: USD 1.600')
st.caption('3º classe: de USD 400 à USD 1.100')
st.text('\n')
st.caption('Por ser uma base de dados do século passado, a informação de sexo contempla apenas Masculino e Feminino.')
st.text('\n')

##Dict de apoio
classes = {'1º classe':1, '2º classe':2, '3º classe':3}

##sidebar
st.sidebar.header('Insira os dados do passageiro:')
form = st.sidebar.form("dados")
idade = form.number_input('Idade', step=1)
sexo = form.selectbox('Sexo',['MASCULINO','FEMININO'])
classe = form.selectbox('Ticket', classes) 
form.markdown('**Quem viaja com você?**')  
form.caption('Total de irmãos e/ou conjuge') 
sibsp = form.number_input('conjuge/irmãos', step=1)
form.caption('Total de pais e/ou filhos') 
parch = form.number_input('pais/filhos', step=1)
submitted = form.form_submit_button(label='Submeter')
st.sidebar.text("\n")
st.sidebar.caption("Feito por Felipe Muros")
st.sidebar.write("[Linkedin](https://www.linkedin.com/in/felipe-muros-48367433/)")
st.sidebar.write("[GitHub](https://github.com/murosfc)")
st.sidebar.write("[Reposiório do projeto](https://github.com/murosfc/titanic-streamlit)")

if submitted:
    #['Age', 'Sex', 'Pclass',  'SibSp', 'Parch']
    classe = classes[classe]
    if sexo == 'MASCULINO':
        sexo = 0
    else:
        sexo = 1
    arry_passageiro = np.array([[idade, sexo, classe, sibsp, parch]])   
    passageiro = pd.DataFrame(arry_passageiro, columns=['Age', 'Sex', 'Pclass',  'SibSp', 'Parch'])   
    st.subheader('Sua chance de sobrevivência seria de: ') 
    taxaSobrevivencia =  getResult(modelo, passageiro)*100    
    st.header('{:.2f}{}'.format(taxaSobrevivencia, '%'))
    st.text('\n')
    st.caption('{}{:.2f}{}'.format('Taxa de precisão o modelo: ', score*100, '%'))

#esconder o footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 