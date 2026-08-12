# Guía del estudiante — Anexo Práctico de IA

**Desarrollo de Sistemas de Inteligencia Artificial** · Trayecto F · 2.º año
ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026

---

## Antes de arrancar

Este anexo acompaña al cuadernillo del Trayecto F. **No lo reemplaza.** El
cuadernillo te dice qué tenés que decidir sobre tu proyecto; el anexo te da las
herramientas para decidirlo con datos en lugar de con intuición.

Trabajás sobre **tu propio proyecto** en todos los ejercicios. Si tu nodo
todavía no genera datos, elegí en el cuaderno A-0 el conjunto de la cohorte que
más se le parezca y usá ese durante todo el anexo. No cambies de conjunto a
mitad de camino: las entregas se encadenan.

---

## Por dónde se empieza

En orden, sin saltear:

1. **Cuaderno 0 — ¿De qué hablamos cuando hablamos de IA?** No tiene fórmulas ni
   requisitos. Si nunca viste nada de inteligencia artificial, este es tu lugar.
2. **Cuaderno 1 — Cómo leer un gráfico.** De acá en adelante casi todo lo que
   tengas que decidir lo vas a decidir mirando un gráfico.
3. **A-0 en adelante**, ya con tus propios datos.

Los dos primeros se dan en clase y se discuten. No los hagas solo en tu casa: la
mitad del valor está en escuchar en qué se equivocaron los demás.

## Cómo se trabaja

### El bucle de cada cuaderno

Cada cuaderno tiene siempre la misma estructura:

1. **Teoría** — para leer, no para saltear. Las decisiones de los ejercicios se
   fundamentan con lo que está ahí.
2. **Demostración** — código ya escrito que corre y muestra el concepto. Leelo,
   cambiale un número, volvé a correrlo. Romperlo a propósito es la forma más
   rápida de entenderlo.
3. **Ejercicios** — con `# TU CÓDIGO ACÁ`. Abajo de cada uno hay una celda de
   verificación.

### La verificación automática

Los ejercicios imprimen `[OK]` o `[REVISAR]` con una pista. **Eso no es tu
nota**: es para que no sigas construyendo sobre algo que quedó mal.

Los ejercicios marcados `[A]` (avanzados) casi nunca tienen verificación
automática, porque piden criterio y no una respuesta única. Son los que más
pesan en la evaluación.

### Los tres niveles

| Marca | Nivel | Qué se espera |
|---|---|---|
| `[B]` | Básico | Aplicación directa de lo que acaba de explicarse |
| `[I]` | Intermedio | Combinar dos ideas, o adaptarlas a tu proyecto |
| `[A]` | Avanzado | Redactar una decisión fundamentada. Sin respuesta única. |

**Todos los `[B]` y `[I]` son obligatorios.** De los `[A]`, son obligatorios los
que están marcados como entrega de un módulo (1.5, A2.5, A4.4, A5.4 y A7.4).

### Reglas de convivencia con Jupyter

- Si algo se comporta raro, **Kernel → Restart & Run All**. Resuelve casi todo.
- Guardá el cuaderno **con las salidas visibles**. Un cuaderno entregado sin
  salidas no se puede corregir.
- No borres las celdas de verificación.
- Podés agregar todas las celdas que quieras para probar cosas.

---

## Qué se entrega

Al final del cuatrimestre entregás **una carpeta** con:

```
apellido_nombre_anexo/
├── notebooks/          los .ipynb ejecutados, con salidas
├── figuras/            las figuras que generaste en A-3
└── INFORME.md          el informe de datos del módulo F-5
```

### Las piezas que se encadenan

Ojo con esto, porque es la parte que más se subestima: **varios ejercicios
producen texto que después va directo a una entrega del cuadernillo.** No los
escribas como si fueran ejercicios sueltos.

| Ejercicio | Alimenta a |
|---|---|
| 0.4 — dónde entraría la IA en tu nodo | G-2, formulación de la tarea |
| 1.5 — un gráfico de la calle, analizado | F-3, criterio de lectura |
| A1.1 — PEAS de tu nodo | F-0, bloque 2 |
| A2.1 — diccionario de variables | **F-1**, entrega completa |
| A2.5 — párrafo "cómo se limpió" | **F-5**, sección 2 del informe |
| A3.2 — figura de tendencia | **F-3** y **F-5**, sección 3 |
| A3.4 — conclusión de una línea | **F-3**, entrega |
| A1.5 + A4.4 — costo de cada error | **G-2**, formulación de la tarea |
| A5.4 — decisión fundamentada | **G-3**, entrega completa |
| A4.4 punto 5 + A5.4 punto 4 | **F-5** y **G-4**, límites declarados |

### El informe final (módulo F-5)

Una página, cuatro secciones, ni una más:

| Sección | De dónde sale |
|---|---|
| **Qué se midió** | tu diccionario de variables (A2.1) |
| **Cómo se limpió** | tu reporte de calidad (A2.5) |
| **Qué se encontró** | tu figura y tu conclusión (A3.2, A3.4) |
| **Qué no se puede afirmar todavía** | tus límites declarados (A4.4, A5.4) |

La cuarta sección es la que separa un informe técnico de un resumen de
actividades. Escribila con honestidad: *"con un solo episodio en el conjunto de
prueba no puedo afirmar que el sistema generalice"* vale mucho más que un
párrafo entusiasta.

---

## Errores que se repiten todos los años

Van sin vueltas, porque saberlos de antemano te ahorra tiempo.

1. **Reportar la exactitud (`accuracy`) y nada más.** Con clases desbalanceadas
   —que es lo normal en detección de fallas— un modelo que dice siempre "no"
   saca 90 %. Siempre acompañá con precisión y sensibilidad.

2. **No calcular el baseline.** Un MAE de 3 % no significa nada hasta que sepas
   cuánto da repetir el último valor conocido. Sin baseline, un número no es un
   resultado.

3. **Partir los datos al azar.** Con series temporales hay que partir por tiempo.
   La partición al azar da métricas mejores y falsas.

4. **Interpolar la variable que se quiere predecir.** Produce métricas
   espectaculares que no miden nada.

5. **Elegir el umbral mirando el conjunto de prueba.** Si ajustás algo mirando la
   prueba, la prueba dejó de ser prueba.

6. **Fijar el criterio de éxito después de ver el resultado.** Decidí *antes* de
   entrenar qué número te alcanzaría, y anotalo.

7. **Escribir conclusiones sobre el gráfico en lugar de sobre el fenómeno.** "Se
   observa una tendencia creciente" no dice nada. "El consumo crece unos 3 W por
   día" sí.

8. **Confundir "el modelo anduvo bien" con "el problema necesita un modelo".** A
   veces la respuesta correcta es que un umbral alcanza, y demostrarlo con
   números es una entrega completa.

---

## Sobre usar asistentes de IA para resolver el anexo

Podés usarlos, y probablemente vayas a hacerlo. Dos advertencias concretas:

**Primera, práctica.** Estos cuadernos están diseñados para que las respuestas
dependan de *tus* datos y *tu* proyecto. Un asistente que no ve tu conjunto de
datos te va a dar código plausible con números inventados. Los ejercicios `[A]`,
que son los que más pesan, piden criterio sobre tu contexto: ahí la ayuda
externa se nota enseguida y no ayuda.

**Segunda, de fondo.** En el cuaderno A-7 vas a entrenar un modelo de lenguaje
con tus propias manos y vas a ver, en chiquito y con total claridad, por qué
produce cosas que suenan bien y son falsas. Ese cuaderno es, entre otras cosas,
la explicación técnica de por qué conviene verificar lo que te devuelve un
asistente antes de entregarlo.

Si usás uno, la regla de la materia es simple: **tenés que poder explicar cada
línea que entregás.** Si no podés, no la entregues.

---

## Si te trabás

En este orden:

1. Releé la sección de teoría del bloque. La pista suele estar ahí.
2. Mirá la pista que imprime la celda de verificación.
3. Probá el mismo código con datos más chicos (`.head(20)`) para ver qué pasa.
4. Preguntale a un compañero de otro grupo.
5. Traelo a clase. Una duda que aparece en varios grupos se explica una vez para
   todos.

Y algo que conviene tener presente: **quedarse trabado un rato es parte del
método, no una señal de que algo anda mal.** El objetivo del anexo no es que
completes celdas, es que puedas defender tus decisiones técnicas frente a
alguien de la cooperativa.
