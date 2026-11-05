---
icon: "{{ project.url }}/common/icons/robot_2.png"
author: Gaspar Corrarello
---

Si no existen los skills que necesitamos, es posible crearlos ya que son archivos de texto plano, con formato markdown.

Incluso es posible crearlos usando otras herramientas de inteligencia artificial, `ChatGPT` incluso tiene skills para la creacion de skills:

El prompt:

```
Investiga si hay skills para que los agentes redacten documentacion con imagenes, 
usando markdown o retype, sino hay, podrias crear un skill para retype? 
investiga todos los componentes para que el agente sepa cuando usar cada componente segun corresponda,
para que la documentacion se pueda leer naturalmente segun su contexto.

[https://retype.com/configuration/page/](https://retype.com/configuration/page/)
[https://retype.com/components/](https://retype.com/components/)
[https://retype.com/feature-log/](https://retype.com/feature-log/)

/create-skill
```

Nos crea el siguiente skill: