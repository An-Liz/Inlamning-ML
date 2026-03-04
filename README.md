# Modellera MNIST


### Inläsning och förståelse av data (Kodblock 1)

Arbetet inleddes med att läsa in MNIST-datasetet via fetch_openml. Datasetet består av 70 000 observationer där varje bild representeras av 784 pixlar (28x28). 
I detta steg kontrollerade jag datans dimensioner och konverterade målvariabeln till numeriskt format.
Det var viktigt för mig att säkerställa att datan var korrekt formaterad innan jag gick vidare, eftersom fel i detta tidiga skede ofta leder till svårtolkade problem senare i processen.

### Uppdelning av data (Kodblock 2)

Därefter delade jag upp datasetet i tränings-, validerings- och testmängd. 
Jag valde att använda tre delmängder för att kunna: träna modeller på träningsdata, välja modell och optimera parametrar på valideringsdata, göra en slutlig, opartisk utvärdering på testdata.
Jag använde stratifiering för att säkerställa att fördelningen av siffrorna 0–9 blev jämn i alla delmängder. 
Detta minskar risken att modellen tränas på en skev klassfördelning. Detta steg var centralt för att arbeta enligt en korrekt maskininlärningsprocess och undvika dataläckage.

### Skalning av data (Kodblock 3)

Eftersom vissa modeller, särskilt Logistic Regression, är känsliga för skalan på indata standardiserade jag features med StandardScaler (se Kodblock 3). 
Jag anpassade skalaren (fit) enbart på träningsdata och applicerade sedan transformationen (transform) på validerings- och testdata. 
Detta gjordes för att undvika dataläckage och säkerställa att ingen information från val- eller testdata påverkade träningen. 
Detta steg tydliggjorde för mig hur viktigt det är att separera träning och utvärdering även i preprocessing-fasen.

### Träning av kandidater (Kodblock 4)

I nästa steg tränade jag fyra olika modeller:
- Logistic Regression
- Random Forest
- Extra Trees
- Voting Classifier

Samtliga modeller tränades enbart på träningsdatan. I detta kodblock genomfördes ingen utvärdering, utan syftet var att skapa tränade kandidater inför jämförelsen. 
Genom att tydligt separera träning och utvärdering blev processen mer strukturerad och lättare att följa.

### Modellutvärdering på valideringsdata (Kodblock 5)

Efter träningen utvärderade jag modellerna på valideringsmängden (se Kodblock 5). Jag använde flera mått:

 - Accuracy
 - Precision (macro)
 - Recall (macro)
 - F1-score (macro och weighted)

Resultatet visade att Extra Trees presterade bäst, följt av Random Forest, medan Logistic Regression presterade lägre. Detta indikerade att problemet är icke-linjärt och att trädmodeller är bättre lämpade för denna typ av bilddata. 
I den inledande modelljämförelsen använde jag en traditionell train/validation-split för att effektivt kunna jämföra flera modeller. När den mest lovande modellen identifierats valde jag att gå vidare med en mer fördjupad optimering.

### Hyperparameteroptimering (Kodblock 6)

När Extra Trees visade sig vara den bäst presterande modellen valde jag att optimera den vidare med hjälp av GridSearchCV (se Kodblock 6).

Jag testade olika kombinationer av:

- antal träd
- max features
- max djup
- insta antal observationer per löv

Optimeringen genomfördes med 3-fold cross validation och utvärderingsmåttet f1_macro, vilket gav en mer robust och stabil uppskattning av modellens generaliseringsförmåga.
Genom att först använda en enkel train/validation-split och därefter cross validation för den utvalda modellen uppnådde jag en god balans mellan beräkningseffektivitet och metodologisk noggrannhet.
Vid deployment användes en praktiskt anpassad version av modellen baserad på optimeringsresultatet, med något begränsad komplexitet för att minska filstorleken.

### Utvärdering av optimerad modell på valideringsdata (Kodblock 7)

Efter optimeringen utvärderade jag den bästa modellen (best_estimator_) på valideringsmängden (se Kodblock 7). Här använde jag både classification report och confusion matrix för att analysera vilka siffror modellen hade svårare att skilja åt.
Detta steg bekräftade att optimeringen gav en mindre men tydlig förbättring jämfört med baseline-versionen.

### Slutlig testutvärdering (Kodblock 8)

När den bästa modellen identifierats tränades den om på hela tränings- och valideringsmängden tillsammans. I samband med att modellen skulle distribueras i en Streamlit-applikation begränsades dock modellens komplexitet genom att sätta ett maxdjup och ett minimikrav på antal observationer per löv. Detta gjordes för att minska modellens filstorlek och möjliggöra publicering via GitHub och Streamlit Cloud. Den slutliga testutvärderingen genomfördes därefter på testmängden, och modellen uppnådde fortsatt hög prestanda (cirka 97–98 % accuracy), vilket visar att justeringen inte påverkade generaliseringsförmågan nämnvärt.

### Sparande av modell (Kodblock 9)

Den slutliga modellen och den anpassade scalern sparades med hjälp av joblib. I samband med detta kontrollerades modellens filstorlek för att säkerställa att den låg under GitHubs gräns för enskilda filer. Efter justering av modellens komplexitet uppgick filstorleken till cirka 67 MB, vilket möjliggjorde publicering och användning i en Streamlit-applikation.

### Avslutande reflektion

Genom att arbeta strukturerat och tydligt separera träning, validering och test har jag följt en korrekt maskininlärningsprocess. Arbetet har tydliggjort skillnaden mellan linjära och icke-linjära modeller samt vikten av att använda flera utvärderingsmått vid modelljämförelse.
En viktig lärdom för mig har varit hur avgörande det är att arbeta metodiskt, särskilt i en notebook-miljö där körordningen påverkar resultatet. Jag har också fått en djupare förståelse för hur hyperparametrar påverkar modellens prestanda och hur en strukturerad optimeringsprocess kan förbättra resultatet på ett kontrollerat sätt.





# Streamlit-applikation


I den sista delen av arbetet utvecklade jag en interaktiv Streamlit-applikation som använder den sparade modellen för att prediktera handritade siffror.
Syftet var att simulera hur modellen fungerar på ny, tidigare osedd data – i detta fall användarens egen handritade siffra.

### Förbehandling av inmatad bild

Eftersom handritade siffror skiljer sig från MNIST-datasetet behövde bilden anpassas så att den liknade träningsdatan så mycket som möjligt. Följande steg implementerades i en separat preprocess()-funktion:

- Konvertering till gråskala
- Invertering (så siffran blir ljus mot mörk bakgrund, som i MNIST)
- Beskärning runt ritad yta
- Skalning så siffran får plats inom 20×20 pixlar
- Centrering med hjälp av viktat center-of-mass
- Placering i en 28×28 canvas
- Transformation med samma scaler som användes under träningen

Genom att återanvända den sparade scalern säkerställdes att modellen fick data i exakt samma format som vid träning, vilket är avgörande för korrekt prediktion.

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