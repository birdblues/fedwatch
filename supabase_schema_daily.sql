-- Create the macro_state_daily table
create table if not exists public.macro_state_daily (
  date date primary key,
  regime_id int not null,
  regime_label text not null,
  stress_flag boolean default false,
  stress_score float8,
  stress_driver text,
  core_cpi_yoy float8,
  core_cpi_yoy_med float8,
  orders_yoy float8,
  orders_mom float8,
  orders_mom_med float8,
  unrate_chg_3m float8,
  yc_10y2y float8,
  hy_oas float8,
  stlfsi4 float8,
  vix float8,
  hmm_score float8,
  prob_regime_0 float8,
  prob_regime_1 float8,
  prob_regime_2 float8,
  prob_regime_3 float8,
  updated_at timestamptz default now()
);

-- Enable Row Level Security (RLS)
alter table public.macro_state_daily enable row level security;

-- Create policy to allow read access to everyone
create policy "Enable read access for all users"
on public.macro_state_daily
for select
using (true);

-- Create policy to allow insert/update access to service_role only
-- This assumes the pipeline uses the service_role key.
create policy "Enable insert for authenticated users only"
on public.macro_state_daily
for insert
with check (auth.role() = 'service_role');

create policy "Enable update for authenticated users only"
on public.macro_state_daily
for update
using (auth.role() = 'service_role');
