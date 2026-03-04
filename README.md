## Streamlit-applikation


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