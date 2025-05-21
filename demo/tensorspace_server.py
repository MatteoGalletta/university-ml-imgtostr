# serve_tensorspace.py
import http.server
import socketserver
import os
from functools import partial

PORT = 8000  # Puoi cambiare la porta se la 8000 è già in uso
SERVE_SUBDIR = "tensorspace_app"  # La sottocartella che contiene i file di TensorSpace

def run_server(port, serve_from_subdir):
    # Percorso completo della directory da servire
    # Si aspetta che SERVE_SUBDIR sia relativa alla posizione di serve_tensorspace.py
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    full_serve_path = os.path.join(current_script_path, serve_from_subdir)

    if not os.path.exists(full_serve_path):
        print(f"Creazione della directory richiesta: {full_serve_path}")
        os.makedirs(full_serve_path)
        print(f"Per favore, colloca il tuo file 'tensorspace_visualization.html' e qualsiasi asset TensorSpace correlato (modelli, ecc.) all'interno della directory '{serve_from_subdir}'.")
    elif not os.path.isfile(os.path.join(full_serve_path, "tensorspace_visualization.html")):
        print(f"ATTENZIONE: 'tensorspace_visualization.html' non trovato in '{full_serve_path}'.")
        print(f"Assicurati che il file HTML sia correttamente posizionato nella sottodirectory '{serve_from_subdir}'.")

    print(f"\nAvvio del server per servire i file dalla sottodirectory: '{full_serve_path}'")
    print(f"Tentativo di avviare il server su http://localhost:{port}")
    print(f"La tua pagina TensorSpace dovrebbe essere accessibile a: http://localhost:{port}/tensorspace.html")
    print("Premi Ctrl+C per fermare il server.")

    # Configura l'handler per servire dalla sottodirectory specificata
    handler_with_subdir = partial(http.server.SimpleHTTPRequestHandler, directory=full_serve_path)

    try:
        with socketserver.TCPServer(("", port), handler_with_subdir) as httpd:
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 98: # Address already in use
             print(f"\nERRORE: La porta {port} è già in uso. Prova a cambiarla nello script 'serve_tensorspace.py'.")
        else:
            print(f"\nERRORE: Impossibile avviare il server: {e}")
    except KeyboardInterrupt:
        print("\nServer fermato dall'utente.")


if __name__ == "__main__":
    run_server(PORT, SERVE_SUBDIR)