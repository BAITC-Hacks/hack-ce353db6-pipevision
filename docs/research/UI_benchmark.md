# UI-бенчмарк: панели прогноза выработки ВЭС и диспетчерские интерфейсы

*Бриф для `ui/app.py` (WindAgent), 23.09.2026. Пользователи: диспетчер и планировщик ВЭС, энергосбытовой аналитик. Ссылки — в разделе 5.*

**Главное.** Лучшие панели показывают прогноз как полосу вероятностей. Рядом на одной оси времени — факт и прошлые выпуски. Рампы вынесены в отдельную ленту с порогами. Основа экрана серая, цветом выделены только отклонения. Все нужные данные у нас уже есть: P10/P50/P90, предыдущий выпуск (его читает `core._previous_forecast`), флаги с порогами в `tools.*_THRESHOLD`, факт SCADA до 31.01.2026. Но панель показывает только текущий выпуск, а флаги выводит отдельно от графика.

## 1. Что общего у лучших панелей

1. **Прогноз показывают полосой, а не линией.** Elia публикует P10 и P90 для каждого выпуска (поля `mostrecentconfidence10/90`, `dayahead11hconfidence10/90`). Vaisala Xweather даёт калиброванные интервалы к каждому прогнозу. В метеограммах ECMWF — 10/25/50/75/90-й перцентили. У Банка Англии fan chart: центр тёмный, к краям светлее, чтобы не создавать ложной точности.
2. **Факт и прогноз стоят на одной оси, граница между ними — «сейчас».** ERCOT каждый час публикует факт за прошлые 48 ч и прогноз на 168 ч. Elia рядом с прогнозами показывает «Measured & upscaled». CAISO Today's Outlook сравнивает факт с прогнозом.
3. **Для одного часа видно несколько выпусков.** У Elia в одном наборе лежат day-ahead 11:00, day-ahead 18:00, most recent и week-ahead. У ENTSO-E — day-ahead, intraday и current. Ревизию видно без отдельного отчёта.
4. **Под разные решения берут разные квантили.** ERCOT публикует STWPF (вероятность превышения 50 %) и WGRPP (80 %). Консервативный квантиль идёт в обязательства, медиана — в оценку.
5. **Рампы — отдельный продукт с порогами.** ERCOT ELRAS смотрит на 6 ч вперёд с шагом 15 мин. Он даёт вероятности рамп для нескольких порогов и трёх окон. Красным выделяются рампы от 1000 МВт за 15 мин, 2000 МВт за 60 мин и 4000 МВт за 180 мин.
6. **Мощность показывают в двух шкалах.** У Elia рядом с МВт стоят `monitoredcapacity` и `loadfactor` (% от мощности).
7. **Экраны выстроены иерархией с переходом вглубь.** В ISA-101 четыре уровня: обзор, установка, узел, тренды и диагностика. В Greenbyte (Power Factors) путь «портфель → парк → турбина» плюс журнал остановов и предупреждений. VestasOnline Business показывает выработку, статусы турбин и алармы.
8. **Основа серая, цвет — только для отклонений (ISA-101).** Фон светло-серый. Для приоритетов алармов оставлена небольшая палитра. Число показывают с контекстом: рядом норма и тренд.

## 2. Ключевые экраны и что берём

**(а) Fan chart P10/P50/P90 + факт + предыдущий выпуск.** *Лидеры:* Elia показывает текущий и дневной выпуски, каждый со своими P10/P90, и факт. ECMWF показывает распределение на каждом шаге времени. *Нам:*
- полоса P10–P90 бледная, того же тона, что P50;
- P50 — единственная насыщенная линия;
- P50 выпуска D−1 — серый пунктир;
- факт — тёмные точки;
- вертикальная линия «момент выпуска», подписи «D+1 · 01.02» у границ суток;
- подпись полосы: «80 %-интервал, по бэктесту покрыто 73–83 % часов» (README §3). Это честнее голой полосы.

Предыдущий выпуск уже читается, его нужно только сохранить в `ForecastResult`. Факт берётся через `wind_agent.evaluate.load_actual_hourly("data/raw/turbine_*.csv")`, он есть до 31.01.2026.

**(б) KPI-панель.** *Лидеры:* Stephen Few советует спарклайны и bullet graph вместо спидометров и один экран. NN/g: главное — крупными числами в верхнем ряду, дальше F-паттерн. ISA-101: число без нормы и тренда почти бесполезно. *Нам:* вместо 9 карточек — 5:
- энергия за 48 ч, МВт·ч (P50 и [P10–P90]);
- сутки D+1 и D+2;
- пик P50: МВт и час;
- риск рампы: максимальный |ΔP50| за час против порога 0,4 и время;
- уверенность: средняя ширина P90−P10 против порога 0,6, словами «высокая / средняя / низкая».

`delta` считать к выпуску D−1 **только по общим часам**: у выпусков D−1 и D общие только сутки D+1, поэтому у энергии за 48 ч дельты нет. Ставить `delta_color="off"`: рост выработки для диспетчера не «хорошо» и не «плохо». Спарклайн P50 передавать через `chart_data`. Решение агента показать бейджем в шапке. Время прогона, запросы API и статус проверок свести в одну строку `st.caption`.

**(в) Лента предупреждений.** *Лидеры:* ELRAS окрашивает только превышение порога. NN/g советует разделять «требует действия» и «к сведению» и беречь внимание от усталости от алертов. Greenbyte и Vestas ведут журнал событий со временем. *Нам:*
- Карточка алерта: уровень (иконка и слово, не только цвет), понятный заголовок, интервал по местному времени, значение и порог, действие. Пример заголовков: «Резкая рампа», «Существенная ревизия», «Окно штиля для ТО». Пример действия: «проверить график выдачи 14:00–16:00».
- Сортировка: сначала warning, потом info; внутри уровня — по времени.
- Те же интервалы подсвечиваются на fan chart.
- `st.toast` — только для warning и только сразу после прогона, не на каждом rerun.
- Счётчик в названии вкладки: «Предупреждения (2)».
- Окно штиля — это возможность, а не проблема, поэтому цвет нейтральный.

**(г) Карта площадки.** *Лидеры:* у Vaisala выработка по рынкам на карте. У Windy — анимация ветра, метеограмма точки и сравнение моделей. VestasOnline показывает схему парка со статусами турбин. *Нам:*
- убрать карту из сайдбара во вкладку «Площадка»;
- турбины выделять цветом только при флаге;
- стрелка ветра на выбранный час: Altair `mark_point(shape="wedge", angle=…)` или `st.pydeck_chart` (pydeck 0.9.3 ставится вместе со Streamlit);
- ссылка на 3D `viz/index.html`.

**(д) Таблицы по часам и суткам с экспортом.** *Лидеры:* ERCOT явно пишет конвенцию часа (hour ending). Elia показывает местное время на экране, а в выгрузке и API — UTC. *Нам:*
- `st.dataframe` с `column_config`: `DatetimeColumn`, `NumberColumn(format="%.2f МВт")`, `ProgressColumn` для P50 в % номинала, колонка флага;
- в суточной таблице — `LineChartColumn` с профилем за 24 ч;
- подпись: «14:00 = среднее за 14:00–15:00» (часовые средние SCADA помечены началом часа);
- CSV в `utf-8-sig`, чтобы Excel открыл кириллицу; местное время и UTC уже есть в контракте.

**(е) «Почему так» (объяснение).** *Лидеры:* в Windy Compare согласие моделей служит мерой уверенности. ECMWF показывает разброс ансамбля. У вендоров, по обзору Wind Systems Magazine, видны модели под консенсусом, интервалы и история точности. *Нам:*
1. Ветер на 100 м: тонкие серые линии GFS, ICON, ECMWF и best_match плюс их среднее. Для этого `session.state["weather"]["frames"]` нужно сохранить в `ForecastResult`.
2. Часы прогноза точками на кривой мощности (ветер → P50). На крутом участке кривой небольшая ошибка ветра даёт большую ошибку мощности, отсюда и широкие полосы.
3. Ревизия по суткам: столбики Δ МВт·ч к выпуску D−1.
4. Климатология: bullet graph по Few — энергия против нормы месяца.
5. Комментарий агента с бейджем проверки фактов.

**(ж) Live и ретроспектива.** *Лидеры:* у ERCOT скользящее окно факта и прогноза, обновление каждый час. Elia показывает most recent рядом с day-ahead. *Нам:*
- бейдж режима в шапке через `st.badge`, например «LIVE · запуск 12:00 UTC+5 · 40 мин назад» или «РЕТРО · выпуск 31.01.2026 23:00»;
- в live — вертикальная линия «сейчас»;
- в ретро — кнопки «← D−1 | D+1 →» и лента 28 выпусков из `outputs/replay_summary.csv` (энергия за 48 ч, решение, флаги), переход по клику через `on_select` у `st.dataframe` или `st.altair_chart`.

**(з) Сравнение с фактом.** *Лидеры:* ERCOT сравнивает факт со STWPF и WGRPP, Elia — measured с выпусками. У Vaisala и enercast есть история прогнозов и точности. *Нам:* вкладка «Факт и точность».
- Для дат с фактом: MAE, смещение, покрытие P10–P90 и столбики остатков (факт − P50). Столбики вне полосы выделяются цветом.
- Для февраля 2026: `st.file_uploader` для файлов организаторов → те же метрики через `wind_agent.evaluate`. Жюри загружает факт и сразу видит точность.
- В expander «Доверие к модели»: MAE 0,175 против 0,210 у кривой мощности и 0,373 у персистентности (README §8).

## 3. Визуальные принципы

- **Серая основа, цвет — для внимания (ISA-101).** Оси, сетка, полоса и прошлый выпуск — серые или приглушённые. Насыщенный цвет только у P50 и у алертов: янтарный — warning, красный — recalculate или проваленные проверки, info без цвета. Уровень кодировать дважды, цветом и иконкой или словом, — для людей с нарушениями цветового зрения. Сейчас полоса и линия яркие `#3b82f6`, ветер бирюзовый: на экране два ярких графика без повода для тревоги.
- **Единицы.** Для диспетчера основные единицы — МВт и МВт·ч; Elia добавляет к ним load factor в %. «Часы номинала» — вторичная единица, в подсказке. Один переключатель единиц на всю страницу (сейчас он только у графика). Число знаков фиксировано: МВт — 1, доля — 2, % — 0. Числа выровнены вправо. Без ложной точности: не `0.173`, а `17 %`.
- **Время.** На экране — местное UTC+5, в выгрузке — местное и UTC, как у Elia. Метки оси вида «Пт 01.02 06:00», подписи суток «D+1». Конвенция часа указана явно. В шапке — время выпуска, запуск модели погоды и возраст прогноза.
- **Плотность.** Главный экран без прокрутки: шапка, 5 KPI, fan chart и топ-3 алерта. Few: один экран. NN/g: KPI → тренд → таблицы. Детали уходят в `st.tabs`. Никаких спидометров и 3D в основной панели.
- **Тема.** Streamlit сам переключает светлую и тёмную тему. Цвета марок Altair брать из двух палитр по `st.context.theme.type` (в 1.64 есть). Для диспетчерской — светло-серая основа, не чисто белая (ISA-101); её можно задать в `.streamlit/config.toml`.
- **Мобильный вид.** `st.columns` на узком экране складываются в столбец. Держать не больше 5 карточек, график высотой около 300 px, главный алерт над графиком. Сценарий: планировщик с телефона проверяет энергию на сутки и алерты; так устроены мобильные приложения Greenbyte.

## 4. План правок `ui/app.py`

Установлены Streamlit 1.64 и Altair 6.3, pydeck идёт вместе со Streamlit, новых зависимостей не нужно. В `pyproject` стоит `streamlit>=1.38`, поэтому новые аргументы (`delta`, `chart_data`, `border`, `on_select`) передавать через существующий `_kw()`, а `st.badge` вызывать через `getattr`. Пороги брать из `core.tools.RAMP_THRESHOLD`, `CALM_LEVEL`, `CALM_MIN_HOURS` и `WIDE_BAND_THRESHOLD`, чтобы панель не разошлась с агентом.

**P0 — первые 40 минут**

| # | Что изменить | Почему | ≈ мин |
|---|---|---|---|
| 1 | Fan chart 2.0: палитра ISA-101 (полоса бледная, P50 тёмная), P50 выпуска D−1 пунктиром, линия «выпуск», подписи D+1/D+2, легенда. В `ui/core.py` добавить `ForecastResult.previous` (DataFrame уже читается) | Ревизию видно на графике, как у Elia; меньше цветового шума | 12 |
| 2 | Флаги на графике: рампы (\|ΔP50\| > 0,4 за час) и окна штиля (≥ 6 ч с P50 ≤ 0,05) пересчитать из `view` и показать через `mark_rect` (янтарный / нейтральный) с подсказкой | Предупреждение привязано ко времени, как в ELRAS | 8 |
| 3 | KPI: 5 карточек `st.metric` с `delta` к D−1 по общим часам (`delta_color="off"`), спарклайнами и МВт·ч как основной единицей. Техкарточки свести в одну строку | Few и NN/g: главное крупно и с контекстом | 10 |
| 4 | Лента алертов: заголовок по коду, интервал, значение/порог, действие; сортировка; `st.toast` для warning после прогона | Оператор видит, что делать, а не код и сообщение | 7 |
| 5 | Каркас: сверху шапка, KPI и график, ниже `st.tabs` «Предупреждения (N)», «Таблицы и выгрузка», «Агент», «Площадка» | Статус на одном экране, детали на следующих уровнях (ISA-101) | 3 |

**P1 — потом, 1–3 ч**

6. Факт на графике и вкладка «Факт и точность»: `load_actual_hourly`, метрики, остатки, `st.file_uploader` для февраля.
7. Таблицы через `column_config` (Datetime, Number с единицами, Progress, LineChart в суточной), CSV в `utf-8-sig`, подпись конвенции часа.
8. Навигация по выпускам: кнопки «← D−1 | D+1 →», лента выпусков из `replay_summary.csv`, бейдж LIVE/РЕТРО с возрастом прогноза.
9. «Почему так»: ансамбль ветра (сохранить `weather.frames`), точки на кривой мощности, ревизия по суткам, bullet graph климатологии.

**P2**

10. Глобальный переключатель единиц и единое форматирование чисел.
11. Вкладка «Площадка»: схема турбин, стрелка ветра на выбранный час, ссылка на 3D.
12. Палитры для светлой и тёмной темы, `.streamlit/config.toml`, проверка на ширине 390 px, `st.fragment(run_every=…)` для live.

## 5. Источники

**Операторы сетей и рынки**
- Elia, Wind-power generation: https://www.elia.be/en/grid-data/power-generation/wind-power-generation
- Elia Open Data, схема ods031 (поля measured / most recent / day-ahead 11AM / P10 / P90, local vs UTC): https://opendata.elia.be/explore/dataset/ods031/ и https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods031
- Elia Open Data, ods086 (near real-time): https://opendata.elia.be/explore/dataset/ods086/
- ERCOT, NP4-732-CD Wind Power Production (48 ч факта + 168 ч прогноза, STWPF 50 % / WGRPP 80 %): https://www.ercot.com/mp/data-products/data-product-details?id=NP4-732-CD
- ERCOT, дашборды STWPF и WGRPP за текущий день: https://www.ercot.com/content/cdr/html/CURRENT_DAYSTWPF.html, https://www.ercot.com/content/cdr/html/CURRENT_DAYWGRPP.html
- ERCOT ELRAS: https://nawindpower.com/ercot-using-new-forecasting-tool-to-prepare-for-wind-variability; Power Operations Bulletin 794 (пороги рамп): http://www.ercot.com/content/wcm/pobs/127910/Power_Operations_Bulletin_794.doc; AMS 2011, продукты ramp-прогноза: https://ams.confex.com/ams/91Annual/webprogram/Paper186686.html
- CAISO Today's Outlook: https://www.caiso.com/todays-outlook
- ENTSO-E, Generation Forecasts for Wind and Solar [14.1.D] (day-ahead / intraday / current): https://transparencyplatform.zendesk.com/hc/en-us/articles/16648445340180-Generation-Forecasts-for-Wind-and-Solar-14-1-D
- 50Hertz, Wind power: https://www.50hertz.com/en/Transparency/GridData/Productiongridfeed-in/Windpower; TenneT, Actual and forecast wind energy feed-in: https://netztransparenz.tennet.eu/electricity-market/transparency-pages/transparency-germany/network-figures/actual-and-forecast-wind-energy-feed-in

**Вендоры прогнозов, SCADA и APM**
- Vaisala Xweather, energy forecasting: https://www.vaisala.com/en/products/renewable-energy/forecaster; https://www.vaisala.com/en/digital-and-data-services/renewable-energy
- enercast: https://www.enercast.de/; Wind Systems Magazine, Wind Power Forecasting: https://www.windsystemsmag.com/wind-power-forecasting/
- Greenbyte / Power Factors: https://apps.apple.com/us/app/greenbyte-wind/id877606940; https://powerfactors.com/parkwind-uses-greenbyte-taxonomy-to-maintain-its-edge-offshore/
- Bazefield: https://bazefield.com/our-products/
- VestasOnline Business SCADA: https://pdf.archiexpo.com/pdf/vestas/vestasonline-business-scada-system/88087-192471.html; https://www.vestas.com/en/energy-solutions/service/digital-services/vestasonline
- Windy, Compare forecast: https://community.windy.com/topic/26304/understanding-the-compare-forecast-feature-in-windy-com
- IEA Wind Task 36, Recommended Practices: https://iea-wind.org/task36/task36-publications/task36-recommended-practices/

**Неопределённость**
- ECMWF, Meteograms (Forecast User Guide 8.1.4): https://confluence.ecmwf.int/display/FUG/Section+8.1.4+Meteograms
- Bank of England, Understanding the fan chart (1998): https://www.bankofengland.co.uk/-/media/boe/files/quarterly-bulletin/1998/the-inflation-report-projections-understanding-the-fan-chart
- Padilla, Kay, Hullman, Uncertainty Visualization (2022): http://space.ucmerced.edu/Downloads/publications/Uncertainty_Visualization_Padilla_Kay_Hullman_2022.pdf

**HMI и дашборды**
- ISA-101 High-Performance HMI: https://ladx.ai/resources/isa-101-hmi-design; https://www.realpars.com/blog/high-performance-hmi
- NN/g, Dashboards: Making Charts and Graphs Easier to Understand: https://www.nngroup.com/articles/dashboards-preattentive/
- NN/g, Indicators, Validations, and Notifications: https://www.nngroup.com/articles/indicators-validations-notifications/; Alert Fatigue in User Interfaces: https://www.nngroup.com/videos/alert-fatigue-user-interfaces/
- Stephen Few, Information Dashboard Design (Analytics Press, 2013); Bullet graph: https://en.wikipedia.org/wiki/Bullet_graph
