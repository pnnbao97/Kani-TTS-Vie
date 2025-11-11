import re

class VietnameseTTSNormalizer:
    """
    A text normalizer for Vietnamese Text-to-Speech systems.
    Converts numbers, dates, units, and special characters into readable Vietnamese text.
    """
    
    def __init__(self):
        self.abbreviations = {
            'tp.hcm': 'thành phố hồ chí minh',
            'tp hcm': 'thành phố hồ chí minh',
            'tphcm': 'thành phố hồ chí minh',
            'hn': 'hà nội',
            'bs': 'bác sĩ',
            'gs': 'giáo sư',
            'pgs': 'phó giáo sư',
            'th.s': 'thạc sĩ',
            'ths': 'thạc sĩ',
            'ts': 'tiến sĩ',
            'vv': 'vân vân',
            '...': 'vân vân',
        }
        
        self.units = {
            'km': 'ki lô mét', 'dm': 'đê xi mét', 'cm': 'xen ti mét',
            'mm': 'mi li mét', 'nm': 'na nô mét', 'µm': 'mic rô mét',
            'μm': 'mic rô mét', 'm': 'mét',
            
            'kg': 'ki lô gam', 'g': 'gam', 'mg': 'mi li gam',
            
            'km²': 'ki lô mét vuông', 'km2': 'ki lô mét vuông',
            'm²': 'mét vuông', 'm2': 'mét vuông',
            'cm²': 'xen ti mét vuông', 'cm2': 'xen ti mét vuông',
            'mm²': 'mi li mét vuông', 'mm2': 'mi li mét vuông',
            'ha': 'héc ta',
            
            'km³': 'ki lô mét khối', 'km3': 'ki lô mét khối',
            'm³': 'mét khối', 'm3': 'mét khối',
            'cm³': 'xen ti mét khối', 'cm3': 'xen ti mét khối',
            'mm³': 'mi li mét khối', 'mm3': 'mi li mét khối',
            'l': 'lít', 'dl': 'đê xi lít', 'ml': 'mi li lít', 'hl': 'héc tô lít',
            
            'v': 'vôn', 'kv': 'ki lô vôn', 'mv': 'mi li vôn',
            'a': 'am pe', 'ma': 'mi li am pe', 'ka': 'ki lô am pe',
            'w': 'oát', 'kw': 'ki lô oát', 'mw': 'mê ga oát', 'gw': 'gi ga oát',
            'kwh': 'ki lô oát giờ', 'mwh': 'mê ga oát giờ', 'wh': 'oát giờ',
            'ω': 'ôm', 'ohm': 'ôm', 'kω': 'ki lô ôm', 'mω': 'mê ga ôm',
            
            'hz': 'héc', 'khz': 'ki lô héc', 'mhz': 'mê ga héc', 'ghz': 'gi ga héc',
            
            'pa': 'pát cal', 'kpa': 'ki lô pát cal', 'mpa': 'mê ga pát cal',
            'bar': 'ba', 'mbar': 'mi li ba', 'atm': 'át mốt phia', 'psi': 'pi ét xai',
            
            'j': 'giun', 'kj': 'ki lô giun', 'mj': 'mê ga giun',
            'cal': 'ca lo', 'kcal': 'ki lô ca lo',
            
            'k': 'ken vin',
            
            'b': 'bai', 'kb': 'ki lô bai', 'mb': 'mê ga bai',
            'gb': 'gi ga bai', 'tb': 'te ra bai', 'pb': 'pê ta bai',
            
            's': 'giây', 'ms': 'mi li giây', 'min': 'phút', 'h': 'giờ',
        }
        
        self.digits = ['không', 'một', 'hai', 'ba', 'bốn', 
                      'năm', 'sáu', 'bảy', 'tám', 'chín']
    
    def normalize(self, text):
        """Main normalization pipeline."""
        text = self._expand_abbreviations(text)
        text = text.lower()
        text = self._normalize_temperature(text)
        text = self._normalize_currency(text)
        text = self._normalize_percentage(text)
        text = self._normalize_units(text)
        text = self._normalize_time(text)
        text = self._normalize_date(text)
        text = self._normalize_phone(text)
        text = self._normalize_numbers(text)
        text = self._number_to_words(text)
        text = self._normalize_special_chars(text)
        text = self._normalize_whitespace(text)
        return text
    
    def _normalize_temperature(self, text):
        """Convert temperature notation to words."""
        text = re.sub(r'-(\d+(?:[.,]\d+)?)\s*°\s*c\b', r'âm \1 độ xê', text, flags=re.IGNORECASE)
        text = re.sub(r'-(\d+(?:[.,]\d+)?)\s*°\s*f\b', r'âm \1 độ ép', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*°\s*c\b', r'\1 độ xê', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*°\s*f\b', r'\1 độ ép', text, flags=re.IGNORECASE)
        text = re.sub(r'°', ' độ ', text)
        return text
    
    def _normalize_currency(self, text):
        """Convert currency notation to words."""
        def decimal_currency(match):
            whole = match.group(1)
            decimal = match.group(2)
            unit = match.group(3)
            decimal_words = ' '.join([self.digits[int(d)] for d in decimal])
            unit_map = {'k': 'nghìn', 'm': 'triệu', 'b': 'tỷ'}
            unit_word = unit_map.get(unit.lower(), unit)
            return f"{whole} phẩy {decimal_words} {unit_word}"
        
        text = re.sub(r'(\d+)[.,](\d+)\s*([kmb])\b', decimal_currency, text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*k\b', r'\1 nghìn', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*m\b', r'\1 triệu', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*b\b', r'\1 tỷ', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*đ\b', r'\1 đồng', text)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*vnd\b', r'\1 đồng', text, flags=re.IGNORECASE)
        text = re.sub(r'\$\s*(\d+(?:[.,]\d+)?)', r'\1 đô la', text)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*\$', r'\1 đô la', text)
        return text
    
    def _normalize_percentage(self, text):
        """Convert percentage to words."""
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*%', r'\1 phần trăm', text)
        return text
    
    def _normalize_units(self, text):
        """Convert measurement units to words."""
        def expand_compound_with_number(match):
            number = match.group(1)
            unit1 = match.group(2).lower()
            unit2 = match.group(3).lower()
            full_unit1 = self.units.get(unit1, unit1)
            full_unit2 = self.units.get(unit2, unit2)
            return f"{number} {full_unit1} trên {full_unit2}"
        
        def expand_compound_without_number(match):
            unit1 = match.group(1).lower()
            unit2 = match.group(2).lower()
            full_unit1 = self.units.get(unit1, unit1)
            full_unit2 = self.units.get(unit2, unit2)
            return f"{full_unit1} trên {full_unit2}"
        
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*([a-zA-Zμµ²³°]+)/([a-zA-Zμµ²³°0-9]+)\b', 
                     expand_compound_with_number, text)
        text = re.sub(r'\b([a-zA-Zμµ²³°]+)/([a-zA-Zμµ²³°0-9]+)\b', 
                     expand_compound_without_number, text)
        
        sorted_units = sorted(self.units.items(), key=lambda x: len(x[0]), reverse=True)
        for unit, full_name in sorted_units:
            pattern = r'(\d+(?:[.,]\d+)?)\s*' + re.escape(unit) + r'\b'
            text = re.sub(pattern, rf'\1 {full_name}', text, flags=re.IGNORECASE)
        
        for unit, full_name in sorted_units:
            if any(c in unit for c in '²³°'):
                pattern = r'\b' + re.escape(unit) + r'\b'
                text = re.sub(pattern, full_name, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_time(self, text):
        """Convert time notation to words."""
        text = re.sub(r'(\d{1,2}):(\d{2}):(\d{2})', r'\1 giờ \2 phút \3 giây', text)
        text = re.sub(r'(\d{1,2}):(\d{2})', r'\1 giờ \2 phút', text)
        text = re.sub(r'(\d{1,2})h(\d{2})', r'\1 giờ \2 phút', text)
        text = re.sub(r'(\d{1,2})h\b', r'\1 giờ', text)
        return text
    
    def _normalize_date(self, text):
        """Convert date notation to words."""
        def date_to_text(match):
            day, month, year = match.groups()
            return f"ngày {day} tháng {month} năm {year}"
        
        def date_iso_to_text(match):
            year, month, day = match.groups()
            return f"ngày {day} tháng {month} năm {year}"
        
        def date_short_year(match):
            day, month, year = match.groups()
            full_year = f"20{year}" if int(year) < 50 else f"19{year}"
            return f"ngày {day} tháng {month} năm {full_year}"
        
        text = re.sub(r'\bngày\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b', 
                     lambda m: f"ngày {m.group(1)} tháng {m.group(2)} năm {m.group(3)}", text)
        text = re.sub(r'\bngày\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})\b', 
                     lambda m: date_short_year(m).replace('ngày ngày', 'ngày'), text)
        text = re.sub(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', date_iso_to_text, text)
        text = re.sub(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b', date_to_text, text)
        text = re.sub(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})\b', date_short_year, text)
        return text
    
    def _normalize_phone(self, text):
        """Convert phone numbers to digit-by-digit reading."""
        def phone_to_text(match):
            phone = match.group(0)
            phone = re.sub(r'[^\d]', '', phone)
            
            if phone.startswith('84') and len(phone) >= 10:
                phone = '0' + phone[2:]
            
            if 10 <= len(phone) <= 11:
                words = [self.digits[int(d)] for d in phone]
                return ' '.join(words) + ' '
            
            return match.group(0)
        
        text = re.sub(r'(\+84|84)[\s\-\.]?\d[\d\s\-\.]{7,}', phone_to_text, text)
        text = re.sub(r'\b0\d[\d\s\-\.]{8,}', phone_to_text, text)
        return text
    
    def _expand_abbreviations(self, text):
        """Expand common abbreviations before lowercasing."""
        sorted_abbr = sorted(self.abbreviations.items(), key=lambda x: len(x[0]), reverse=True)
        for abbr, full in sorted_abbr:
            pattern = r'\b' + re.escape(abbr) + r'\.?'
            text = re.sub(pattern + r'\b', ' ' + full + ' ', text, flags=re.IGNORECASE)
        return text
    
    def _normalize_numbers(self, text):
        """Normalize number formats (thousand separators and decimals)."""
        text = re.sub(r'(\d{1,3})(?:\.(\d{3}))+\b', 
                     lambda m: m.group(0).replace('.', ''), text)
        
        def decimal_to_words(match):
            whole = match.group(1)
            decimal = match.group(2)
            decimal_words = ' '.join([self.digits[int(d)] for d in decimal])
            return f"{whole} phẩy {decimal_words}"
        
        text = re.sub(r'(\d+),(\d+)', decimal_to_words, text)
        text = re.sub(r'(\d+)\.(\d{1,2})\b', decimal_to_words, text)
        return text
    
    def _read_two_digits(self, n):
        """Read two-digit numbers in Vietnamese."""
        if n < 10:
            return self.digits[n]
        elif n == 10:
            return "mười"
        elif n < 20:
            if n == 15:
                return "mười lăm"
            return f"mười {self.digits[n % 10]}"
        else:
            tens = n // 10
            ones = n % 10
            if ones == 0:
                return f"{self.digits[tens]} mươi"
            elif ones == 1:
                return f"{self.digits[tens]} mươi một"
            elif ones == 5:
                return f"{self.digits[tens]} mươi lăm"
            else:
                return f"{self.digits[tens]} mươi {self.digits[ones]}"
    
    def _read_three_digits(self, n):
        """Read three-digit numbers in Vietnamese."""
        if n < 100:
            return self._read_two_digits(n)
        
        hundreds = n // 100
        remainder = n % 100
        result = f"{self.digits[hundreds]} trăm"
        
        if remainder == 0:
            return result
        elif remainder < 10:
            result += f" lẻ {self.digits[remainder]}"
        else:
            result += f" {self._read_two_digits(remainder)}"
        
        return result
    
    def _convert_number_to_words(self, num):
        """Convert a number to Vietnamese words."""
        if num == 0:
            return "không"
        
        if num < 0:
            return f"âm {self._convert_number_to_words(-num)}"
        
        if num >= 1000000000:
            billion = num // 1000000000
            remainder = num % 1000000000
            result = f"{self._read_three_digits(billion)} tỷ"
            if remainder > 0:
                result += f" {self._convert_number_to_words(remainder)}"
            return result
        
        elif num >= 1000000:
            million = num // 1000000
            remainder = num % 1000000
            result = f"{self._read_three_digits(million)} triệu"
            if remainder > 0:
                result += f" {self._convert_number_to_words(remainder)}"
            return result
        
        elif num >= 1000:
            thousand = num // 1000
            remainder = num % 1000
            result = f"{self._read_three_digits(thousand)} nghìn"
            if remainder > 0:
                if remainder < 100:
                    result += f" không trăm {self._read_two_digits(remainder)}"
                else:
                    result += f" {self._read_three_digits(remainder)}"
            return result
        
        else:
            return self._read_three_digits(num)
    
    def _number_to_words(self, text):
        """Convert all remaining numbers to words."""
        def convert_number(match):
            num = int(match.group(0))
            return self._convert_number_to_words(num)
        
        text = re.sub(r'\b\d+\b', convert_number, text)
        return text
    
    def _normalize_special_chars(self, text):
        """Handle special characters."""
        text = text.replace('&', ' và ')
        text = text.replace('+', ' cộng ')
        text = text.replace('=', ' bằng ')
        text = text.replace('#', ' thăng ')
        text = re.sub(r'[\[\]\(\)\{\}]', ' ', text)
        text = re.sub(r'\s+[-–—]+\s+', ' ', text)
        text = re.sub(r'\.{2,}', ' ', text)
        text = re.sub(r'\s+\.\s+', ' ', text)
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ.,!?;:@]', ' ', text)
        return text
    
    def _normalize_whitespace(self, text):
        """Normalize whitespace."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text


if __name__ == "__main__":
    normalizer = VietnameseTTSNormalizer()
    
    test_texts = [
        "Giá 2.500.000đ (giảm 50%), mua trước 14h30 ngày 15/12/2025",
        "Liên hệ: 0912-345-678 hoặc email@example.com",
        "Tốc độ 120km/h, trọng lượng 75kg",
        "Nhiệt độ 36,5°C, độ ẩm 80%",
        "TP.HCM HN, PGS.TS.BS. Nguyễn Văn A",
        "Số pi = 3,14159",
        "Giá trị tăng 2.5M, đạt 10B",
        "Nhiệt độ -15°C vào mùa đông",
        "Điện áp 220V, công suất 2.5kW, tần số 50Hz",
        "Tôi đi lấy l nước về nhà",
        "Cần 5l nước cho công thức này",
        "Vận tốc ánh sáng 299792km/s",
        "Mật độ dân số 450 người/km2",
        "Công suất 100 W/m2",
        "Hôm nay 2025-01-15",
        "Gọi +84 912 345 678",
        "Nhiệt độ 25°C lúc 14:30:45",
        "Ngày 15/12/25",
        "Giá 3.140.159",
    ]
    
    print("=" * 80)
    print("VIETNAMESE TTS NORMALIZATION TEST")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\n📝 Input: {text}")
        normalized = normalizer.normalize(text)
        print(f"🎵 Output: {normalized}")
        print("-" * 80)