COMO HACER FUNCIONAR EL BOT 

1.INSTALAR LAS LIBRERIAS

Para hacer funcionar el bot, se necesitaran las siguientes librerias:
openai
langchain 
chromadb
docx2txt
pypdf
streamlit
tiktoken
unstructured
python-pptx
pdfminer
pdf2image
tesseract
unstructured_inference
pdfminer.six

Si es tu primera vez usandolo, deberas instalarlas, para instalarlas todas al instante, abre tu terminal y escribe el siguiente comando:

pip install requirements.txt

De no correr directamente utiliza

pip install -r requirements.txt

En cuanto la libreria tesseract, no es obligatorio, pero si no la instalas, recuerda comentarla en el archivo "Bot_layla_sphere.py"
y comentar las opciones que avisan que puede leer imagenes, ya que esas funciones no funcionaran 
sin tesseract.



2. HACER FUNCIONAR EL BOT

2.1 Visualizar el front del bot
Para lograr que el bot funcione,deberas ir al archivo "Bot_layla_sphere.py" y dentro de el, 
abre tu terminal(ctrl+ñ) y despues escribir el siguiente comando:

streamlit run main.py

Esto en cuyo caso el nombre del archivo no cambie conforme pase el tiempo.

2.2 Hacer hablar el bot

En el bot "Browse files" que se encuentra en la pagina creada por streamlit, deberas escoger el
documento que deseas que analice el bot 

despues de cargar el documentos ten en cuenta los datos que se encuentran abajo:

chunk_size: Este componente es un número de entrada (input) que permite al usuario especificar 
el tamaño de los fragmentos (chunks) en los que se dividirá el contenido del archivo. El usuario
puede ajustar este valor utilizando los controles proporcionados. Los parámetros min_value y 
max_value limitan el rango de valores permitidos, y value establece el valor predeterminado en 
512. Cuando el usuario cambia este valor, se invoca la función clear_history utilizando el 
parámetro on_change.

k: Similar al componente anterior, este es otro número de entrada que permite al usuario 
especificar el valor de "k". Este valor se utiliza en la recuperación de respuestas. El usuario
puede ajustar este valor en el rango de 1 a 20, y el valor predeterminado es 3. Al cambiar este
valor, también se invoca la función clear_history utilizando on_change.

add_data: Este componente es un botón con el texto "Add Data". El usuario puede hacer clic en 
este botón después de cargar un archivo para confirmar la adición de datos. Cuando se hace clic
en este botón, se invoca la función clear_history. Esto se utiliza para borrar el historial de
conversación antes de realizar nuevas preguntas.


Para finalizar, cuando le des clic al boton add data, espera a que cargue la info y despues
escribe lo que le quieras preguntar al bot respecto al documento seleccionado.

###COMENTARIOS ADICIONALES###

El bot tiene documentos adicionales en la carpeta, que son para localizar paths, otras versiones del bot, estas las encuentras en la carpeta "Zona de pruebas",

Ademas cuenta con una datbase que se necesitaba para lanzarlo como demo se encuentra en la carpeta "Database".


Las cosas faltantes para lanzar el bot era estructurar una database dentro del bot sin necesidad de darle documentos manualmente al bot para analizarlo, para que al final se creara con fastapi el bot dentro de ella, para poder empezar a construir el front del bot

###Instalaciones#####

pip install langchain_openai

pip install langchain --upgrade

pip install wikipedia

pip3 install streamlit

pip install pymupdf


####Levantar servicio con docker####

docker compose build 
docker compose up -d

