@echo off
setlocal EnableDelayedExpansion

echo === Bat dau xu ly ===
echo.

:: Tao file tam chua danh sach ten goc
dir /b *_s1.json *_s2.json *_s3.json 2>nul | findstr /r /c:"_s[123]\.json$" > temp_files.txt

if not exist temp_files.txt (
    echo Khong tim thay file nao dang *_s1.json / *_s2.json / *_s3.json
    goto :end
)

:: Lay danh sach ten goc duy nhat
for /f "delims=" %%f in (temp_files.txt) do (
    set "name=%%f"
    set "base=!name:~0,-8!"
    echo !base!>> temp_bases.txt
)

sort temp_bases.txt /unique > temp_unique.txt

for /f "delims=" %%b in (temp_unique.txt) do (
    set "base=%%b"
    set "count=0"
    set "files="

    :: Thu thap cac file cua nhom
    if exist "!base!_s1.json" (
        set /a count+=1
        set "files=!files! !base!_s1.json"
    )
    if exist "!base!_s2.json" (
        set /a count+=1
        set "files=!files! !base!_s2.json"
    )
    if exist "!base!_s3.json" (
        set /a count+=1
        set "files=!files! !base!_s3.json"
    )

    if !count! gtr 0 (
        echo Nhom: !base!  (!count! file)

        :: Chon ngau nhien
        set /a "rand=!random! %% !count!"
        set "idx=0"
        set "keep="

        for %%f in (!files!) do (
            if !idx! equ !rand! (
                set "keep=%%f"
            )
            set /a idx+=1
        )

        echo   - Giu lai: !keep!

        :: Doi ten
        if not "!keep!"=="!base!.json" (
            ren "!keep!" "!base!.json"
            echo   - Doi ten thanh: !base!.json
        ) else (
            echo   - Da dung ten roi
        )

        :: Xoa cac file con lai
        for %%f in (!files!) do (
            if not "%%f"=="!keep!" (
                if exist "%%f" (
                    echo   - Xoa: %%f
                    del "%%f"
                )
            )
        )
        echo.
    )
)

:end
del temp_files.txt temp_bases.txt temp_unique.txt 2>nul

echo === Hoan tat ===
echo Cac file con lai:
dir /b *.json 2>nul
echo.
pause