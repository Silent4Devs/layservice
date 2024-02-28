import psycopg2
import streamlit as st

# Función para buscar en la base de datos PostgreSQL por nombre de usuario y pregunta
def buscar_usuario_en_bd(nombre_usuario, pregunta):
    conn = psycopg2.connect(
        dbname="nuevabd76",
        user="postgres",
        password="",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM empleados WHERE name = %s", (nombre_usuario,))
    registro = cursor.fetchone()
    
    # Obtener el área del empleado si la pregunta es sobre el área
    if pregunta.lower() == "cual es su area":
        cursor.execute("SELECT area FROM areas WHERE id = %s", (registro[8],))  # Suponiendo que el área está en la columna 7 de la tabla de empleados
        area = cursor.fetchone()
        conn.close()
        return area[0] if area else None
    
     # Obtener la sede del empleado si la pregunta es sobre la sede
    if pregunta.lower() == "cual es tu sede":
        cursor.execute("SELECT sede FROM sedes WHERE id = %s", (registro[16],))  # Suponiendo que la sede está en la columna 9 de la tabla de empleados
        sede = cursor.fetchone()
        conn.close()
        return sede[0] if sede else None
    

      # Obtener la sede del empleado si la pregunta es sobre la sede
    if pregunta.lower() == "quien es  su supervisor":
        cursor.execute("SELECT name FROM empleados WHERE id = %s", (registro[7],))  # Suponiendo que la sede está en la columna 9 de la tabla de empleados
        sede = cursor.fetchone()
        conn.close()
        return sede[0] if sede else None
    
    conn.close()
    
    # Analizar la pregunta y devolver la información correspondiente
    if pregunta.lower() == "cual es su correo":
        return registro[6] if registro else None
    elif pregunta.lower() == "cual es su estatus":
        return registro[5] if registro else None
    elif pregunta.lower() == "cual es su informacion":
        return registro if registro else None
    else:
        return None

# Interfaz de usuario con Streamlit
st.title("ChatBoot Silent4business")
    
# Campo de entrada de texto para el nombre de usuario
nombre_usuario = st.text_input("Ingrese el nombre de usuario:")
    
# Campo de entrada de texto para preguntar sobre la base de datos
pregunta_bd = st.text_input("Haga una pregunta acerca del usuario:")
    
# Botón para realizar la búsqueda
if st.button("Buscar"):
    # Realizar la búsqueda en la base de datos si se hizo una pregunta
    if pregunta_bd:
        resultado = buscar_usuario_en_bd(nombre_usuario, pregunta_bd)
        # Mostrar el resultado de la pregunta en la interfaz de usuario
        if resultado:
            if pregunta_bd.lower() == "cual es su informacion":
                st.write("Información del empleado:")
                for campo, valor in zip(["ID", "Nombre", "Foto", "Puesto", "Antiguedad", "Estatus", "Email"], resultado):
                    st.write(f"{campo}: {valor}")
            else:
                st.write("Respuesta:", resultado)
        else:
            st.warning("No se encontró respuesta para la pregunta en la base de datos.")
