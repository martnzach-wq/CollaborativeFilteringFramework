# Strategy dashboard

`regsim.app.strategy_dash` (decision D66) shows a local climate strategy on the SD model: the levers as multiples m
of the stated level, live model runs, targets and margins, the response and climate bands kept apart, lever
contributions, scenario comparison and the assumptions behind each lever. It is city-independent; a city supplies a
`StrategyAdapter` (`regsim.app.strategy_adapter`).

## Adapter

| Attribute or method | Content |
|---|---|
| `title`, `sectors` | page title and lever sectors, per language (`{"de": ..., "en": ...}`) |
| `levers` | `LeverSpec`: key (backcast group), sector, label, bounds, effect class (D63), confidence, the strategy level in natural units, ids of the data-gap register |
| `presets` | named multipliers and options, e.g. the backcast solutions (D64) |
| `targets` | `TargetSpec`: quantity, year, value, change against a year or level, bound (max / min), hard or reported |
| `pathways`, `stacks` | series of the main charts and stacked compositions at milestone years |
| `options` | city choices (climate member, driver variant, timing) |
| `response_bands`, `climate_series` | which option values form the response band; which series carry a climate band |
| `references` | reference pathways (e.g. the run without strategy) |
| `data_gaps` | register table (id, item, current value, importance, informant) |
| `run(multipliers, options)` | results of one scenario (index year); `options["band"]` selects a response band |
| `run_many`, `run_climate_band`, `level_text` | parallel runs, the climate band (long table), the lever in natural units at m |

## Views

1. **Overview:** status of the hard targets (icon and text), main pathways with target markers (filled: hard, open:
   reported) and the reference, emissions by sector.
2. **Targets and margins:** per target and year the central value and margin, the margins in each response band and
   the lowest margin over the climate members; met within the solver tolerance.
3. **Bands:** response band as a shaded envelope on the pathways, climate band (min–max, median) on the series that
   depend on the climate; the two are never merged.
4. **Lever contributions:** one-at-a-time change of each hard target when a lever is set to its reference.
5. **Compare:** the current setting against up to three presets, pathways and multipliers.
6. **Assumptions and data gaps:** the register entries linked to the levers.

Each lever carries its effect class and confidence as labels, so that a result shows which part the model estimates
and which part is an input. Colours follow the reference palette of the dataviz method (fixed categorical order,
status colours with icon and text). Languages: German and English, switchable.

## Running

`create_strategy_app(adapter, lang="de").run()` for local use; `app.server` is the WSGI application for a server
(one worker process, the adapter holds the model; access protection in front of it).
