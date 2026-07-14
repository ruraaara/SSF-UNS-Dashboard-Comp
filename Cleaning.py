

import pandas as pd
import os
os.chdir('SSDC UNS')  # atau path lengkap foldernya

# 1. LOAD 
company = pd.read_csv('company.csv', encoding='utf-8-sig')
status_student = pd.read_csv('status_student.csv', encoding='utf-8-sig', sep=';')
student_all = pd.read_csv('student_all.csv', encoding='utf-8-sig')
talent_request = pd.read_csv('talent_request.csv', encoding='utf-8-sig')
tracking_company = pd.read_csv('tracking_company.csv', encoding='utf-8-sig')
tracking_student = pd.read_csv('tracking_student.csv', encoding='utf-8-sig')

# 2. STANDARDISASI TIPE KEY

company['id_company'] = company['id_company'].astype(str).str.strip()
talent_request['id_talent_req'] = talent_request['id_talent_req'].astype(str).str.strip()
talent_request['id_company'] = talent_request['id_company'].astype(str).str.strip()
tracking_company['id_tracking_company'] = tracking_company['id_tracking_company'].astype(str).str.strip()
tracking_company['id_talent_req'] = tracking_company['id_talent_req'].astype(str).str.strip()
tracking_company['id_company'] = tracking_company['id_company'].astype(str).str.strip()
tracking_student['id_tracking_student'] = tracking_student['id_tracking_student'].astype(str).str.strip()
tracking_student['id_tracking_company'] = tracking_student['id_tracking_company'].astype(str).str.strip()

# NIM konsisten sebagai string di 3 tabel yang pakai
student_all['NIM'] = student_all['NIM'].astype(str).str.strip()
status_student['NIM'] = status_student['NIM'].astype(str).str.strip()
tracking_student['NIM'] = tracking_student['NIM'].astype(str).str.strip()
status_student['id_status'] = status_student['id_status'].astype(str).str.strip()



# 3. STANDARDISASI TANGGAL

company['created_at'] = pd.to_datetime(company['created_at'], format='%Y-%m-%d')
talent_request['request_date'] = pd.to_datetime(talent_request['request_date'], format='%Y-%m-%d')
tracking_student['last_update'] = pd.to_datetime(tracking_student['last_update'], format='%Y-%m-%d')

tracking_company['request_date'] = pd.to_datetime(tracking_company['request_date'], format='%d/%m/%Y')
tracking_company['send_date'] = pd.to_datetime(tracking_company['send_date'], format='%d/%m/%Y')  # NaT tetap NaT, aman
status_student['sync_date'] = pd.to_datetime(status_student['sync_date'], format='%d/%m/%Y')


# 4. PERBAIKAN NOMOR TELEPON

def fix_phone(series):
    s = series.astype(str).str.strip()
    return s.where(s.str.startswith('0'), '0' + s)

student_all['hp'] = fix_phone(student_all['hp'])
company['pic_phone'] = fix_phone(company['pic_phone'])
talent_request['no_whatsapp'] = fix_phone(talent_request['no_whatsapp'])
status_student['no_whatsapp'] = fix_phone(status_student['no_whatsapp'])


# 5. MISSING VALUE
tracking_company['status_pengiriman'] = tracking_company['send_date'].notna().map(
    {True: 'Sudah Dikirim', False: 'Belum Dikirim'}
)

# 6. VALIDASI

assert company['id_company'].duplicated().sum() == 0
assert student_all['NIM'].duplicated().sum() == 0
assert status_student['IPK'].between(0, 4).all()
assert (~talent_request['id_company'].isin(company['id_company'])).sum() == 0
assert (~tracking_company['id_talent_req'].isin(talent_request['id_talent_req'])).sum() == 0
assert (~tracking_student['id_tracking_company'].isin(tracking_company['id_tracking_company'])).sum() == 0


# 7. EXPORT

company.to_csv('cleaned_company.csv', index=False)
status_student.to_csv('cleaned_status_student.csv', index=False)
student_all.to_csv('cleaned_student_all.csv', index=False)
talent_request.to_csv('cleaned_talent_request.csv', index=False)
tracking_company.to_csv('cleaned_tracking_company.csv', index=False)
tracking_student.to_csv('cleaned_tracking_student.csv', index=False)

print("Cleaning selesai. Ringkasan:")
print(f"- company: {len(company)} baris")
print(f"- status_student: {len(status_student)} baris")
print(f"- student_all: {len(student_all)} baris")
print(f"- talent_request: {len(talent_request)} baris")
print(f"- tracking_company: {len(tracking_company)} baris ({(tracking_company['status_pengiriman']=='Belum Dikirim').sum()} belum dikirim)")
print(f"- tracking_student: {len(tracking_student)} baris")
print("Semua validasi FK & rentang nilai: PASSED")