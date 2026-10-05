# Despliegue local de la interfaz grafica (src/interfaz.py).
#
#   make desplegar     levanta (o reinicia) la interfaz y abre el navegador
#   make estado        que esta arriba: modelo vLLM e interfaz
#   make detener       baja la interfaz
#
# El modelo (Qwen3-VL-Embedding) ya esta desplegado en vLLM: este Makefile solo
# consulta si responde, no lo arranca ni lo detiene.
#
# Variables, en la linea de comandos:  make desplegar PUERTO=8502 ABRIR=no
#   PUERTO          puerto de la interfaz                     (8501)
#   DIRECCION       interfaz de red; 0.0.0.0 para la red local (localhost)
#   URL_EMBEDDING   servidor vLLM del embedding               (http://localhost:8001)
#   ABRIR           si / no: abrir el navegador al desplegar  (si)

PUERTO        ?= 8501
DIRECCION     ?= localhost
URL_EMBEDDING ?= http://localhost:8001
ABRIR         ?= si

VENV      := .venv
PY        := $(VENV)/bin/python
STREAMLIT := $(abspath $(VENV)/bin/streamlit)
RUN       := $(abspath .run)
PID       := $(RUN)/interfaz.pid
LOG       := $(RUN)/interfaz.log
URL_FILE  := $(RUN)/interfaz.url
HOST_APP  := $(if $(filter 0.0.0.0,$(DIRECCION)),localhost,$(DIRECCION))
URL_APP   := http://$(HOST_APP):$(PUERTO)

# URL con la que se desplego la instancia que esta corriendo (puede ser otro PUERTO)
URL_ACTUAL = $$(cat $(URL_FILE) 2>/dev/null || echo $(URL_APP))
CORRIENDO  = { [ -f $(PID) ] && kill -0 $$(cat $(PID)) 2>/dev/null; }

.DEFAULT_GOAL := ayuda
.PHONY: ayuda instalar modelos desplegar detener estado logs abrir

ayuda: ## muestra esta ayuda
	@echo "Interfaz: angulo entre dos embeddings"
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*## "} {printf "  make %-10s %s\n", $$1, $$2}'

instalar: ## crea .venv e instala las dependencias
	@test -d $(VENV) || uv venv --python 3.12 $(VENV)
	VIRTUAL_ENV=$(VENV) uv pip install --torch-backend=auto \
	  torch torchvision transformers sentence-transformers \
	  scikit-learn matplotlib numpy pandas reportlab pymupdf pillow \
	  streamlit plotly

modelos: ## verifica que el modelo de embeddings desplegado en vLLM responda
	@if modelo=$$(curl -sf -m 5 $(URL_EMBEDDING)/v1/models | $(PY) -c \
	    'import sys, json; print(json.load(sys.stdin)["data"][0]["root"])' 2>/dev/null); then \
	  printf "  ✓ %-10s %s  (%s)\n" embedding $$modelo $(URL_EMBEDDING); \
	else \
	  printf "  ✗ %-10s no responde en %s\n" embedding $(URL_EMBEDDING); \
	  echo "  Sin el modelo de embeddings la interfaz no puede calcular nada."; exit 1; \
	fi

desplegar: modelos ## levanta (o reinicia) la interfaz y abre el navegador
	@test -x $(STREAMLIT) || { echo "  Falta el entorno: corre 'make instalar'."; exit 1; }
	@# puerto ocupado: solo se acepta si es la instancia que se va a reiniciar
	@if ! $(PY) -c 'import socket, sys; sys.exit(socket.socket().connect_ex(("$(HOST_APP)", $(PUERTO))) == 0)'; then \
	  if $(CORRIENDO) && [ "$(URL_ACTUAL)" = "$(URL_APP)" ]; then :; else \
	    echo "  El puerto $(PUERTO) esta ocupado por otro proceso. Prueba: make desplegar PUERTO=8502"; exit 1; \
	  fi; \
	fi
	@if $(CORRIENDO); then $(MAKE) --no-print-directory detener; fi
	@mkdir -p $(RUN)
	@cd src; \
	URL_EMBEDDING=$(URL_EMBEDDING) nohup $(STREAMLIT) run interfaz.py \
	  --server.port $(PUERTO) --server.address $(DIRECCION) --server.headless true \
	  > $(LOG) 2>&1 & \
	echo $$! > $(PID); echo $(URL_APP) > $(URL_FILE)
	@for i in $$(seq 1 60); do \
	  curl -sf -m 2 $(URL_APP)/_stcore/health > /dev/null && break; \
	  if ! $(CORRIENDO); then \
	    echo "  La interfaz no arranco. Ultimas lineas del log:"; tail -n 20 $(LOG); \
	    rm -f $(PID) $(URL_FILE); exit 1; \
	  fi; \
	  sleep 0.5; \
	done
	@curl -sf -m 2 $(URL_APP)/_stcore/health > /dev/null || \
	  { echo "  La interfaz no respondio a tiempo; revisa: make logs"; exit 1; }
	@echo "  ✓ interfaz  $(URL_APP)  (PID $$(cat $(PID)), log en .run/interfaz.log)"
	@if [ "$(ABRIR)" = si ]; then $(MAKE) --no-print-directory abrir; fi

detener: ## baja la interfaz
	@if $(CORRIENDO); then \
	  kill $$(cat $(PID)); \
	  for i in $$(seq 1 20); do $(CORRIENDO) || break; sleep 0.25; done; \
	  if $(CORRIENDO); then kill -9 $$(cat $(PID)); fi; \
	  echo "  ✓ interfaz detenida (PID $$(cat $(PID)))"; \
	else \
	  echo "  · la interfaz no estaba corriendo"; \
	fi; \
	rm -f $(PID) $(URL_FILE)

estado: ## muestra que esta arriba: modelo vLLM e interfaz
	@$(MAKE) --no-print-directory modelos || true
	@if $(CORRIENDO) && curl -sf -m 2 $(URL_ACTUAL)/_stcore/health > /dev/null; then \
	  printf "  ✓ %-10s %s  (PID %s)\n" interfaz $(URL_ACTUAL) $$(cat $(PID)); \
	else \
	  printf "  ✗ %-10s no esta corriendo (make desplegar)\n" interfaz; \
	fi

logs: ## sigue el log de la interfaz (Ctrl+C para salir)
	@test -f $(LOG) || { echo "  No hay log todavia: make desplegar"; exit 1; }
	@tail -n 40 -f $(LOG)

abrir: ## abre la interfaz en el navegador
	@if command -v xdg-open > /dev/null && [ -n "$$DISPLAY$$WAYLAND_DISPLAY" ]; then \
	  xdg-open $(URL_ACTUAL) > /dev/null 2>&1 & \
	  echo "  → abriendo $(URL_ACTUAL)"; \
	else \
	  echo "  Abre en el navegador: $(URL_ACTUAL)"; \
	fi
