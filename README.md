# MNIST-modell


### Inläsning och förståelse av data (Kodblock 1)

Jag började arbetet inleddes med att läsa in MNIST-datasetet via fetch_openml. Datasetet består av 70 000 observationer där varje bild representeras av 784 pixlar (28x28). 
I detta steg kontrollerade jag datans dimensioner och konverterade målvariabeln till numeriskt format.
Jag ville säkerställa att datan var korrekt formaterad innan jag gick vidare.

### Uppdelning av data (Kodblock 2)

Därefter delade jag upp datasetet i tränings-, validerings- och testmängd. 
Jag valde att använda tre delmängder för att kunna: träna modeller på träningsdata, välja modell och optimera parametrar på valideringsdata, göra en slutlig, opartisk utvärdering på testdata.
Jag använde stratifiering för att säkerställa att fördelningen av siffrorna 0–9 blev jämn i alla delmängder. 
Detta för att minska risken att modellen tränas på en skev klassfördelning.

### Skalning av data (Kodblock 3)

Eftersom vissa modeller, särskilt Logistic Regression, är känsliga för skalan på indata standardiserade jag features med StandardScaler. 
Jag anpassade skalaren (fit) enbart på träningsdata och applicerade sedan transformationen (transform) på validerings- och testdata. 

### Träning av kandidater (Kodblock 4)

I nästa steg tränade jag fyra olika modeller:
- Logistic Regression
- Random Forest
- Extra Trees
- Voting Classifier

Samtliga modeller tränades enbart på träningsdatan. I detta kodblock genomfördes ingen utvärdering, utan syftet var att skapa tränade kandidater inför jämförelsen. 
Genom att tydligt separera träning och utvärdering blev processen mer strukturerad och lättare att följa för mig.

### Modellutvärdering på valideringsdata (Kodblock 5)

Efter träningen utvärderade jag modellerna på valideringsmängden. Jag använde flera mått:
 - Accuracy
 - Precision
 - Recall
 - F1-score

Resultatet visade att Extra Trees presterade bäst, följt av Random Forest, medan Logistic Regression presterade lägre. Detta indikerade att problemet är icke-linjärt och att trädmodeller är bättre lämpade för denna typ av bilddata. 
I den inledande modelljämförelsen använde jag en traditionell train/validation-split för att effektivt kunna jämföra flera modeller. När den mest lovande modellen identifierats valde jag att gå vidare med en mer fördjupad optimering.

### Hyperparameteroptimering (Kodblock 6)

När Extra Trees visade sig vara den bäst presterande modellen valde jag att optimera den vidare med hjälp av GridSearchCV.

Jag testade olika kombinationer av:
- antal träd
- max features
- max djup
- minsta antal observationer per löv

Optimeringen genomfördes med 3-fold cross validation och utvärderingsmåttet f1_macro.
Genom att först använda en enkel train/validation-split och därefter cross validation för den utvalda modellen försökte jag uppnå en god balans mellan beräkningseffektivitet och metodologisk noggrannhet.
Vid deployment användes en praktiskt anpassad version av modellen baserad på optimeringsresultatet, med något begränsad komplexitet för att minska filstorleken.

### Utvärdering av optimerad modell på valideringsdata (Kodblock 7)

Efter hyperparameteroptimeringen valde jag den modell som presterade bäst i grid search. Den testades sedan på valideringsdatan för att se hur den fungerade på ny data. För att analysera resultatet använde jag en classification report och en confusion matrix.
De visade både modellens totala noggrannhet och vilka siffror som ibland blandades ihop. Resultatet visade en liten men tydlig förbättring jämfört med baseline-modellen.

### Slutlig testutvärdering (Kodblock 8)

När den bästa modellen hade identifierats genom optimeringen tränades den om på hela tränings- och valideringsmängden tillsammans för att utnyttja så mycket data som möjligt. Därefter genomfördes en slutlig utvärdering på den separata testmängden för att uppskatta modellens generaliseringsförmåga på helt osedd data. Resultatet visade att modellen uppnådde en fortsatt hög prestanda, med en accuracy på cirka 95 %, vilket var i linje med resultaten från tidigare valideringssteg.

I samband med att modellen skulle distribueras i en Streamlit-applikation behövde modellens komplexitet därefter begränsas genom att sätta ett maxdjup och ett minimikrav på antal observationer per löv. Detta gjordes för att minska modellens filstorlek och möjliggöra publicering via GitHub och Streamlit Cloud.

Jag är medveten om att testmängden i strikt metodik endast bör användas en gång i slutet av processen. I detta fall användes testdatan även för att verifiera att den förenklade modellen, som behövdes för deployment, inte försämrade prestandan. Eftersom justeringen gjordes av praktiska distributionsskäl och inte för ytterligare optimering av modellen bedömde jag att detta var rimligt inom ramen för projektet.

### Sparande av modell (Kodblock 9)

Den slutliga modellen och den anpassade scalern sparades med hjälp av joblib. I samband med detta kontrollerades modellens filstorlek för att säkerställa att den låg under GitHubs gräns för enskilda filer. Efter justering av modellens komplexitet uppgick filstorleken till cirka 67 MB, vilket möjliggjorde publicering och användning i en Streamlit-applikation.

### Avslutande reflektion

Genom att arbeta strukturerat och tydligt separera träning, validering och test har jag följt en korrekt maskininlärningsprocess, även om jag i slutändan blev tvungen att frångå den något. Arbetet har tydliggjort skillnaden mellan linjära och icke-linjära modeller samt vikten av att använda flera utvärderingsmått vid modelljämförelse.
En viktig lärdom för mig har varit hur avgörande det är att arbeta metodiskt, särskilt i en notebook-miljö där körordningen påverkar resultatet. Jag har också fått en djupare förståelse för hur hyperparametrar påverkar modellens prestanda och hur en strukturerad optimeringsprocess kan förbättra resultatet på ett kontrollerat sätt.





# Streamlit-applikation
https://inlamning-ml-annab.streamlit.app/


Jag gick sen på den 2 av uppgiftern, att utveckla en Streamlit-applikation som använder den sparade modellen för att prediktera handritade siffror.
Syftet var att simulera hur modellen fungerar på ny, tidigare osedd data – i detta fall användarens egen handritade siffra.

### Förbehandling av inmatad bild

Eftersom handritade siffror skiljer sig från MNIST-datasetet behövde bilden anpassas så att den liknade träningsdatan så mycket som möjligt. Jag implementerade Följande steg i en separat preprocess()-funktion:

- Konvertering till gråskala
- Invertering (så siffran blir ljus mot mörk bakgrund, som i MNIST)
- Beskärning runt ritad yta
- Skalning så siffran får plats inom 20×20 pixlar
- Centrering med hjälp av viktat center-of-mass
- Placering i en 28×28 canvas
- Transformation med samma scaler som användes under träningen

Genom att återanvända den sparade scalern säkerställdes att modellen fick data i exakt samma format som vid träning, vilket är visade sig avgörande för korrekt prediktion.

### Förbättring av handritad input

Under utvecklingen upptäcktes att tunna konturer ibland gav osäkra prediktioner. Därför lades en valbar funktion till för att förstärka streck med hjälp av ett MaxFilter. Detta gör att konturer kan fyllas något och bättre likna MNIST-siffror.
Användaren kan även justera pennbredd inom ett begränsat intervall (8–12 pixlar) för att minska variation mellan olika ritstilar.

### Användargränssnitt

Gränssnittet byggdes med tydlig struktur:

- Ritområde till vänster
- Prediktion och sannolikheter till höger
- Inställningar i en sidopanel
- En tydlig primärknapp för prediktion
- Separata knappar för att rensa canvas och återställa inställningar

Layouten utformades för att göra arbetsflödet intuitivt: rita → prediktera → justera vid behov.

### Reflektion

Arbetet med Streamlit visade hur känslig en maskininlärningsmodell kan vara för skillnader mellan träningsdata och verklig inmatning. Små variationer i tjocklek, position eller kontrast kan påverka resultatet.
Genom att successivt förbättra preprocessingen och anpassa användargränssnittet kunde modellen användas stabilt även på handritad data, utan att ändra själva modellen.