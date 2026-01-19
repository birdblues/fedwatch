-- Create the macro_regime table
create table if not exists public.macro_regime (
  date date primary key,
  regime_id int not null,
  regime_label text not null,
  stress_flag boolean default false,
  updated_at timestamptz default now()
);

-- Enable Row Level Security (RLS)
alter table public.macro_regime enable row level security;

-- Create policy to allow read access to everyone
create policy "Enable read access for all users"
on public.macro_regime
for select
using (true);

-- Create policy to allow insert/update access to service_role only
-- Note: Service role bypasses RLS by default, but explicit policies can be good practice or needed for authenticated users.
-- For simple scripts using service_role key, they bypass RLS automatically.
-- If you access via 'anon' key for writes (not recommended), you'd need a policy here.
-- This policy allows full access to authenticated users (modify as needed for stricter control)
create policy "Enable insert for authenticated users only"
on public.macro_regime
for insert
with check (auth.role() = 'service_role');

create policy "Enable update for authenticated users only"
on public.macro_regime
for update
using (auth.role() = 'service_role');
