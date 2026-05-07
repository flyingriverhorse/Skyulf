import React, { useRef } from 'react';
import { FileUp, Upload, X } from 'lucide-react';

interface FileUploaderProps {
    file: File | null;
    onFileChange: (file: File | null) => void;
}

/** CSV/Parquet file picker with selected-file pill and remove button. */
export const FileUploader: React.FC<FileUploaderProps> = ({ file, onFileChange }) => {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            onFileChange(e.target.files[0]);
        }
    };

    return (
        <div className="flex items-center gap-2 shrink-0">
            <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                onChange={handleChange}
                accept=".csv,.parquet"
            />
            {file ? (
                <div className="flex items-center gap-1.5 px-3 py-2.5 border border-gray-200 dark:border-slate-600 rounded-md text-sm bg-gray-50 dark:bg-slate-900">
                    <FileUp size={14} className="text-green-500 shrink-0" />
                    <span className="text-slate-700 dark:text-slate-300 truncate max-w-[180px]">{file.name}</span>
                    <button
                        type="button"
                        onClick={() => onFileChange(null)}
                        className="text-gray-400 hover:text-red-500 transition-colors ml-1"
                    >
                        <X size={14} />
                    </button>
                </div>
            ) : (
                <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center gap-2 px-3 py-2.5 border border-dashed border-gray-300 dark:border-slate-600 rounded-md text-sm text-gray-500 dark:text-gray-400 hover:border-blue-400 hover:text-blue-500 dark:hover:border-blue-500 transition-colors"
                >
                    <Upload size={15} />
                    Upload CSV / Parquet
                </button>
            )}
        </div>
    );
};
